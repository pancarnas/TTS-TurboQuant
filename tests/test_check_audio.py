"""GPU/audio-free tests for tools/check_audio.py (filename parse + aggregation)."""

from __future__ import annotations

from tools.check_audio import _pair_diff_rate, parse_wav_name, summarize


def test_parse_wav_name_handles_underscored_groups():
    d = parse_wav_name("qwen_seedtts_en_7_sampling_s0_t0.9_K4_V2_rw=24.wav")
    assert d["group"] == "seedtts_en" and d["idx"] == 7
    assert d["config"] == "K4V2@24" and (d["kb"], d["vb"], d["rw"]) == (4, 2, 24)
    d2 = parse_wav_name("/x/qwen_libritts_long_12_sampling_s0_t0.9_K4_V4_rw=0.wav")
    assert d2["group"] == "libritts_long" and d2["config"] == "K4V4@0"


def test_parse_wav_name_optional_protected_layers_segment():
    # old names (no _pl=) keep parsing with pl=None and an unchanged config label
    d = parse_wav_name("qwen_seedtts_en_7_sampling_s0_t0.9_K4_V2_rw=24.wav")
    assert d["pl"] is None and d["config"] == "K4V2@24"
    # new names carry pl and it distinguishes the config
    d2 = parse_wav_name("qwen_seedtts_en_7_sampling_s0_t0.9_K4_V2_rw=24_pl=0.wav")
    assert d2["pl"] == 0 and d2["config"] == "K4V2@24 pl0"
    assert (d2["kb"], d2["vb"], d2["rw"]) == (4, 2, 24)
    d3 = parse_wav_name("qwen_libritts_long_12_sampling_s0_t0.9_K4_V4_rw=0_pl=2.wav")
    assert d3["group"] == "libritts_long" and d3["pl"] == 2


def test_parse_wav_name_rejects_nonmatching():
    assert parse_wav_name("something_else.wav") is None
    assert parse_wav_name("qwen_x_1_greedy.wav") is None


def _st(md5, dur=2.0, rms=0.1, n=48000, ok=True):
    return {
        "ok": ok,
        "md5": md5,
        "sr": 24000,
        "dur": dur,
        "rms": rms,
        "peak": 0.5,
        "n": n,
    }


def test_summarize_flags_health_and_identity():
    sents = {
        # all-identical sentence (compression NOT applied) + a silent + a good one
        ("g", 0): {"K4V4@24": _st("aaa"), "K4V4@0": _st("aaa")},
        ("g", 1): {"K4V4@24": _st("bbb"), "K4V4@0": _st("ccc", rms=1e-6)},  # silent rw0
        ("g", 2): {
            "K4V4@24": _st("ddd"),
            "K4V4@0": _st("eee", dur=120.0),
        },  # runaway rw0
    }
    s = summarize(sents, silence_rms=1e-4, runaway_s=60.0)
    assert s["n_sentences"] == 3 and s["n_wavs"] == 6
    assert s["multi_config_sentences"] == 3
    assert s["all_identical_sentences"] == 1  # only sentence 0
    assert s["silent"] == 1 and s["runaway"] == 1 and s["bad"] == 0


def test_summarize_counts_unreadable():
    sents = {("g", 0): {"K4V4@24": {"ok": False, "err": "boom"}, "K4V4@0": _st("z")}}
    s = summarize(sents)
    assert s["bad"] == 1 and s["multi_config_sentences"] == 1


def test_pair_diff_rate():
    sents = {
        ("g", 0): {"K4V4@24": _st("a"), "K4V4@0": _st("a")},  # identical
        ("g", 1): {"K4V4@24": _st("a"), "K4V4@0": _st("b")},  # differ
    }
    differ, have = _pair_diff_rate(sents, "K4V4@24", "K4V4@0")
    assert (differ, have) == (1, 2)
