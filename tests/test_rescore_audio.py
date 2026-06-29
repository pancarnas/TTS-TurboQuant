"""Tests for tools/rescore_audio.py — the pure, GPU-free parts.

Transcription is injected, so these run without Whisper or audio files.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tools.rescore_audio import (
    resolve_wav,
    score_trials,
    to_16k,
    wav_name,
)


def _row(**kw):
    base = {
        "group": "seedtts_en",
        "idx": 7,
        "arm": "sampling",
        "seed": 0,
        "temperature": 0.9,
        "config": "K4/V4 rw=24",
        "cer": 0.99,
    }
    base.update(kw)
    return pd.Series(base)


def test_wav_name_matches_benchmark_pattern():
    # benchmark writes: qwen_{group}_{idx}_{arm}_s{seed}_t{temp}_{safe}.wav
    assert wav_name(_row()) == "qwen_seedtts_en_7_sampling_s0_t0.9_K4_V4_rw=24.wav"


def test_wav_name_baseline_and_no_temperature():
    assert (
        wav_name(_row(config="baseline (no TQ)"))
        == "qwen_seedtts_en_7_sampling_s0_t0.9_baseline_(no_TQ).wav"
    )
    assert wav_name(_row(temperature=np.nan)).endswith("_sampling_s0_K4_V4_rw=24.wav")


def test_resolve_wav_exact_then_glob_fallback(tmp_path):
    d = tmp_path / "outputs"
    d.mkdir()
    # exact match
    (d / "qwen_seedtts_en_7_sampling_s0_t0.9_K4_V4_rw=24.wav").write_bytes(b"x")
    assert resolve_wav(_row(), str(d)) is not None
    # different temp formatting on disk -> glob fallback still finds it
    (d / "qwen_seedtts_en_8_sampling_s0_t1.0_K4_V4_rw=24.wav").write_bytes(b"x")
    assert resolve_wav(_row(idx=8, temperature=0.123), str(d)) is not None
    # genuinely absent
    assert resolve_wav(_row(idx=999), str(d)) is None


def test_to_16k_resamples_and_monos():
    sr = 24000
    stereo = np.zeros((sr, 2), dtype=np.float32)  # 1 s stereo
    out = to_16k(stereo, sr)
    assert out.ndim == 1
    assert out.dtype == np.float32
    assert abs(len(out) - 16000) <= 2  # ~1 s at 16 kHz
    # already-16k passes through unchanged in length
    mono16 = np.zeros(16000, dtype=np.float32)
    assert len(to_16k(mono16, 16000)) == 16000


def test_score_trials_replaces_cer_and_keeps_original(tmp_path):
    d = tmp_path / "outputs"
    d.mkdir()
    for r in (_row(idx=0, config="baseline (no TQ)"), _row(idx=1)):
        (d / wav_name(r)).write_bytes(b"x")
    df = pd.DataFrame(
        [_row(idx=0, config="baseline (no TQ)", cer=0.80), _row(idx=1, cer=0.99)]
    )
    text_lookup = {("seedtts_en", 0): "hello world", ("seedtts_en", 1): "hello world"}
    # idx 0 transcribes perfectly (CER 0); idx 1 is garbage (CER high)
    fake = {
        wav_name(_row(idx=0, config="baseline (no TQ)")): "hello world",
        wav_name(_row(idx=1)): "zzz qqq",
    }

    def transcribe(path):
        import os

        return fake[os.path.basename(path)]

    def score(target, hyp):
        return 0.0 if hyp == target else 1.0

    out = score_trials(df, text_lookup, str(d), "qwen", transcribe, score)
    assert list(out["cer_orig"]) == [0.80, 0.99]  # original preserved
    assert list(out["cer"]) == [0.0, 1.0]  # corrected
    assert list(out["transcript"]) == ["hello world", "zzz qqq"]


def test_score_trials_marks_missing_as_nan(tmp_path):
    d = tmp_path / "outputs"
    d.mkdir()
    df = pd.DataFrame([_row(idx=42)])
    out = score_trials(df, {}, str(d), "qwen", lambda p: "x", lambda t, h: 1.0)
    assert np.isnan(out["cer"].iloc[0])
    assert out["cer_orig"].iloc[0] == 0.99
