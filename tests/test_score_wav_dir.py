"""Tests for tools/score_wav_dir.py — the pure, GPU-free parts.

Whisper/WavLM/soundfile are injected into score_entries, so these run without
audio deps; the module's heavy imports are all lazy.
"""

from __future__ import annotations

import csv

from tools.score_wav_dir import (
    FP16_PL,
    attach_baseline,
    collect_entries,
    done_keys,
    entry_key,
    parse_wav_name,
    score_entries,
)


def test_parse_wav_name_quantized():
    e = parse_wav_name("qwen_seedtts_en_7_sampling_s0_t0.9_K4_V3_rw=24_pl=0.wav")
    assert e == {
        "group": "seedtts_en",
        "idx": 7,
        "seed": 0,
        "temperature": 0.9,
        "key_bits": 4,
        "value_bits": 3,
        "rw": 24,
        "pl": 0,
        "config": "K4V3@24",
        "wav": "qwen_seedtts_en_7_sampling_s0_t0.9_K4_V3_rw=24_pl=0.wav",
    }


def test_parse_wav_name_fp16():
    e = parse_wav_name("qwen_librispeech_pc_12_sampling_s0_t0.9_fp16.wav")
    assert e["config"] == "fp16" and e["pl"] == FP16_PL
    assert e["group"] == "librispeech_pc" and e["idx"] == 12
    assert e["key_bits"] == "" and e["rw"] == ""


def test_parse_wav_name_rejects_foreign():
    # old real-benchmark formats and non-experiment files must not match
    assert parse_wav_name("qwen_g_0_sampling_s0_t0.9_baseline_(no_TQ).wav") is None
    assert parse_wav_name("qwen_g_0_sampling_s0_t0.9_K4_V4_rw=24.wav") is None  # no pl
    assert parse_wav_name("notes.txt") is None


def test_parse_wav_name_vallex_variants():
    e = parse_wav_name("vallex_seedtts_en_7_sampling_s0_K4V4@64.wav")
    assert e["group"] == "seedtts_en" and e["idx"] == 7 and e["arm"] == "sampling"
    assert e["config"] == "K4V4@64" and e["key_bits"] == 4 and e["rw"] == 64
    assert e["pl"] == FP16_PL and e["temperature"] == ""

    nar = parse_wav_name("vallex_smoke_1_sampling_s0_K4V4@64-nar.wav")
    assert nar["config"] == "K4V4@64-nar"

    both = parse_wav_name("vallex_smoke_1_greedy_s2_t0.9_K3V3@128-both.wav")
    assert both["config"] == "K3V3@128-both" and both["arm"] == "greedy"
    assert both["seed"] == 2 and both["temperature"] == 0.9

    fp = parse_wav_name("vallex_librispeech_pc_12_sampling_s0_fp16.wav")
    assert fp["config"] == "fp16" and fp["group"] == "librispeech_pc"
    assert fp["pl"] == FP16_PL


def test_attach_baseline_vallex_is_arm_scoped(tmp_path):
    _touch(
        tmp_path,
        [
            "vallex_g_0_sampling_s0_fp16.wav",
            "vallex_g_0_sampling_s0_K4V4@64-nar.wav",
            "vallex_g_0_greedy_s0_K4V4@64.wav",  # no greedy fp16 -> missing
        ],
    )
    entries, _ = collect_entries(str(tmp_path))
    missing = attach_baseline(entries)
    assert missing == 1
    by = {(e.get("arm"), e["config"]): e for e in entries}
    assert by[("sampling", "K4V4@64-nar")]["baseline_path"].endswith("_fp16.wav")
    assert by[("greedy", "K4V4@64")]["baseline_path"] is None


def test_score_entries_wer_column(tmp_path):
    _touch(tmp_path, ["vallex_g_0_sampling_s0_fp16.wav"])
    entries, _ = collect_entries(str(tmp_path))
    attach_baseline(entries)
    rows = []
    score_entries(
        entries,
        text_lookup={("g", 0): "hello"},
        transcribe=lambda p: "hello",
        score=lambda r, h: 0.0,
        embed=lambda p: (1.0, 0.0),
        wav_duration=lambda p: 1.0,
        emit=rows.append,
        score_wer=lambda r, h: 0.5,
    )
    assert rows[0]["wer"] == 0.5 and rows[0]["cer"] == 0.0
    assert rows[0]["arm"] == "sampling"


def _touch(d, names):
    for n in names:
        (d / n).write_bytes(b"x")


def test_collect_entries_sorts_by_sentence_and_counts_foreign(tmp_path):
    _touch(
        tmp_path,
        [
            "qwen_g_1_sampling_s0_t0.9_K4_V4_rw=24_pl=0.wav",
            "qwen_g_0_sampling_s0_t0.9_fp16.wav",
            "qwen_g_0_sampling_s0_t0.9_K4_V4_rw=24_pl=0.wav",
            "qwen_g_0_sampling_s0_t0.9_baseline_(no_TQ).wav",  # foreign
            "readme.md",  # not a wav: ignored silently
        ],
    )
    entries, foreign = collect_entries(str(tmp_path))
    assert foreign == 1
    # sentence-adjacent: both idx=0 configs before idx=1
    assert [(e["idx"], e["config"]) for e in entries] == [
        (0, "K4V4@24"),
        (0, "fp16"),
        (1, "K4V4@24"),
    ]
    assert all(e["path"].endswith(e["wav"]) for e in entries)


def test_attach_baseline_pairs_and_counts_missing(tmp_path):
    _touch(
        tmp_path,
        [
            "qwen_g_0_sampling_s0_t0.9_fp16.wav",
            "qwen_g_0_sampling_s0_t0.9_K4_V4_rw=24_pl=0.wav",
            "qwen_g_1_sampling_s0_t0.9_K4_V4_rw=24_pl=0.wav",  # no fp16 for idx=1
        ],
    )
    entries, _ = collect_entries(str(tmp_path))
    missing = attach_baseline(entries)
    assert missing == 1
    by = {(e["idx"], e["config"]): e for e in entries}
    assert by[(0, "fp16")]["baseline_path"] is None  # baseline rows: no self-pairing
    assert by[(0, "K4V4@24")]["baseline_path"].endswith("_fp16.wav")
    assert by[(1, "K4V4@24")]["baseline_path"] is None


def test_done_keys_roundtrip_with_entry_key(tmp_path):
    entries = [
        parse_wav_name("qwen_g_0_sampling_s0_t0.9_fp16.wav"),
        parse_wav_name("qwen_g_0_sampling_s0_t0.9_K4_V4_rw=24_pl=0.wav"),
    ]
    out = tmp_path / "scores.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["group", "idx", "config", "pl"])
        w.writeheader()
        w.writerow({"group": "g", "idx": 0, "config": "fp16", "pl": FP16_PL})
    done = done_keys(str(out))
    assert entry_key(entries[0]) in done
    assert entry_key(entries[1]) not in done
    assert done_keys(str(tmp_path / "absent.csv")) == set()


def test_score_entries_metrics_and_baseline_embed_reuse(tmp_path):
    _touch(
        tmp_path,
        [
            "qwen_g_0_sampling_s0_t0.9_fp16.wav",
            "qwen_g_0_sampling_s0_t0.9_K4_V4_rw=24_pl=0.wav",
            "qwen_g_0_sampling_s0_t0.9_K4_V3_rw=24_pl=0.wav",
            "qwen_g_1_sampling_s0_t0.9_fp16.wav",  # no ground-truth text -> skipped
        ],
    )
    entries, _ = collect_entries(str(tmp_path))
    attach_baseline(entries)

    embed_calls = []

    def embed(path):
        embed_calls.append(path)
        # normalized 2-d vectors: baseline (1,0); compressed (0.8, 0.6) -> cos 0.8
        return (1.0, 0.0) if path.endswith("_fp16.wav") else (0.8, 0.6)

    rows = []
    score_entries(
        entries,
        text_lookup={("g", 0): "hello"},
        transcribe=lambda path: "hello there",
        score=lambda ref, hyp: 0.25,
        embed=embed,
        wav_duration=lambda path: 1.5,
        emit=rows.append,
    )

    assert [(r["config"], r["spk_sim"]) for r in rows] == [
        ("K4V3@24", 0.8),
        ("K4V4@24", 0.8),
        ("fp16", ""),  # baseline: CER only
    ]
    assert all(r["cer"] == 0.25 and r["dur_s"] == 1.5 for r in rows)
    assert all(r["transcript"] == "hello there" for r in rows)
    # baseline embedded once for the sentence, not once per compressed config
    assert sum(p.endswith("_fp16.wav") for p in embed_calls) == 1
