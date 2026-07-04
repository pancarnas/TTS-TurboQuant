"""Verify generated compressed audio: are configs actually different, and is
inference/saving correct? GPU-free — reads only the saved wavs.

Filenames are ``qwen_<group>_<idx>_sampling_s<seed>_t<temp>_K<kb>_V<vb>_rw=<rw>.wav``
with an optional trailing ``_pl=<protected_layers>`` segment (newer runs).
For each (group, idx) it compares the wavs across configs by content hash + basic
stats (duration, RMS, peak), then reports:

  * SAVING/INFERENCE health — files that fail to load, are empty, silent, or
    runaway-long (a sign of collapse);
  * COMPRESSION applied — how often the configs are byte-IDENTICAL. If (almost)
    every sentence has all configs identical, compression is NOT reaching
    generation (the track_only bug) — the configs should differ, especially the
    aggressive ones (K4V2, K3V3, K4V4@rw0) vs the mild K4V4@rw24.

Run:
  python tools/check_audio.py --audio-dir models/Qwen3-TTS/benchmarks/outputs
  python tools/check_audio.py --audio-dir results/audio_real --verbose
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import os
import re

import numpy as np
import soundfile as sf

_NAME_RE = re.compile(
    r"qwen_(?P<group>.+)_(?P<idx>\d+)_sampling_s(?P<seed>\d+)_t(?P<temp>[\d.]+)"
    r"_K(?P<kb>\d+)_V(?P<vb>\d+)_rw=(?P<rw>\d+)(?:_pl=(?P<pl>\d+))?\.wav$"
)

# VALL-E-X benchmark naming (benchmark_vallex_real.py --configs runs):
#   vallex_<group>_<idx>_<arm>_s<seed>[_t<temp>]_<config>.wav
# where <config> is 'fp16' or a K<kb>V<vb>@<rw> label.
_VALLEX_RE = re.compile(
    r"vallex_(?P<group>.+)_(?P<idx>\d+)_(?P<arm>[a-z]+)_s(?P<seed>\d+)"
    r"(?:_t(?P<temp>[\d.]+))?"
    r"_(?:(?P<fp>fp16)|K(?P<kb>\d+)V(?P<vb>\d+)@(?P<rw>\d+))\.wav$"
)


def parse_wav_name(name: str) -> dict | None:
    """Parse a saved-wav filename into its (group, idx, config) fields, or None."""
    base = os.path.basename(name)
    m = _NAME_RE.search(base)
    if m:
        d = m.groupdict()
        pl = int(d["pl"]) if d["pl"] is not None else None
        # pl joins the config label so runs at different protection settings in
        # the same dir never compare as "the same config".
        config = f"K{d['kb']}V{d['vb']}@{d['rw']}" + (
            f" pl{pl}" if pl is not None else ""
        )
        return {
            "group": d["group"],
            "idx": int(d["idx"]),
            "config": config,
            "kb": int(d["kb"]),
            "vb": int(d["vb"]),
            "rw": int(d["rw"]),
            "pl": pl,
        }
    m = _VALLEX_RE.search(base)
    if m:
        d = m.groupdict()
        if d["fp"]:
            config, kb, vb, rw = "fp16", None, None, None
        else:
            kb, vb, rw = int(d["kb"]), int(d["vb"]), int(d["rw"])
            config = f"K{kb}V{vb}@{rw}"
        return {
            "group": d["group"],
            "idx": int(d["idx"]),
            "config": config,
            "kb": kb,
            "vb": vb,
            "rw": rw,
            "pl": None,
        }
    return None


def wav_stats(path: str) -> dict:
    """Content hash + duration/RMS/peak. ``ok=False`` if the file won't load."""
    try:
        wav, sr = sf.read(path)
    except Exception as exc:  # noqa: BLE001 - report unreadable files, don't crash
        return {"ok": False, "err": str(exc)}
    if getattr(wav, "ndim", 1) > 1:
        wav = wav.mean(axis=1)
    wav = np.asarray(wav, dtype="float64")
    n = int(wav.shape[0])
    md5 = hashlib.md5(open(path, "rb").read()).hexdigest()[:10]
    return {
        "ok": True,
        "md5": md5,
        "sr": int(sr),
        "dur": n / sr if sr else 0.0,
        "rms": float(np.sqrt(np.mean(wav**2))) if n else 0.0,
        "peak": float(np.abs(wav).max()) if n else 0.0,
        "n": n,
    }


def summarize(
    sent_stats: dict, silence_rms: float = 1e-4, runaway_s: float = 60.0
) -> dict:
    """Aggregate per-sentence {config: stats} into health + identity metrics."""
    configs: set[str] = set()
    n_wavs = bad = empty = silent = runaway = 0
    all_identical = multi_config = 0
    for _key, per_cfg in sent_stats.items():
        configs |= set(per_cfg)
        md5s = set()
        for st in per_cfg.values():
            n_wavs += 1
            if not st.get("ok"):
                bad += 1
                continue
            if st["n"] == 0:
                empty += 1
            if st["rms"] < silence_rms:
                silent += 1
            if st["dur"] > runaway_s:
                runaway += 1
            md5s.add(st["md5"])
        if len(per_cfg) > 1:
            multi_config += 1
            if len(md5s) == 1:  # every config produced byte-identical audio
                all_identical += 1
    return {
        "n_sentences": len(sent_stats),
        "n_wavs": n_wavs,
        "configs": sorted(configs),
        "bad": bad,
        "empty": empty,
        "silent": silent,
        "runaway": runaway,
        "multi_config_sentences": multi_config,
        "all_identical_sentences": all_identical,
    }


def _pair_diff_rate(sent_stats, mild="K4V4@24", aggressive="K4V4@0") -> tuple:
    """Fraction of sentences where mild vs aggressive config audio differs."""
    have = differ = 0
    for per_cfg in sent_stats.values():
        a, b = per_cfg.get(mild), per_cfg.get(aggressive)
        if a and b and a.get("ok") and b.get("ok"):
            have += 1
            differ += int(a["md5"] != b["md5"])
    return differ, have


def load_sentences(audio_dir: str) -> dict:
    files = [
        f for f in glob.glob(os.path.join(audio_dir, "*.wav")) if parse_wav_name(f)
    ]
    sents: dict = {}
    for f in files:
        meta = parse_wav_name(f)
        sents.setdefault((meta["group"], meta["idx"]), {})[meta["config"]] = wav_stats(
            f
        )
    return sents


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", default="models/Qwen3-TTS/benchmarks/outputs")
    parser.add_argument(
        "--verbose", action="store_true", help="Per-sentence breakdown."
    )
    parser.add_argument("--silence-rms", type=float, default=1e-4)
    parser.add_argument("--runaway-s", type=float, default=60.0)
    args = parser.parse_args()

    sents = load_sentences(args.audio_dir)
    if not sents:
        raise SystemExit(f"no matching wavs under {args.audio_dir}")
    s = summarize(sents, args.silence_rms, args.runaway_s)

    print(f"\n== audio check: {args.audio_dir} ==")
    print(f"sentences={s['n_sentences']}  wavs={s['n_wavs']}  configs={s['configs']}")
    print("\n-- saving / inference health --")
    print(f"  unreadable/corrupt : {s['bad']}")
    print(f"  empty (0 samples)  : {s['empty']}")
    print(f"  silent (rms<{args.silence_rms:g}) : {s['silent']}")
    print(f"  runaway (>{args.runaway_s:g}s)   : {s['runaway']}")

    print("\n-- compression applied? --")
    mc, ai = s["multi_config_sentences"], s["all_identical_sentences"]
    print(f"  sentences with >1 config : {mc}")
    print(
        f"  ALL configs byte-identical: {ai}/{mc}"
        + ("  <-- compression NOT applied!" if mc and ai == mc else "")
    )
    for mild, aggr in (
        ("K4V4@24", "K4V4@0"),
        ("K4V4@24", "K4V2@24"),
        ("K4V4@24", "K3V3@24"),
    ):
        d, h = _pair_diff_rate(sents, mild, aggr)
        if h:
            print(
                f"  {mild} vs {aggr}: differ {d}/{h}"
                + ("  (good)" if d else "  <-- identical, suspicious")
            )
    # When an fp16 baseline is present (VALL-E-X naming), diff it against every
    # quantized config — identical audio means the lossy path never ran.
    if "fp16" in s["configs"]:
        for cfg in s["configs"]:
            if cfg == "fp16":
                continue
            d, h = _pair_diff_rate(sents, "fp16", cfg)
            if h:
                print(
                    f"  fp16 vs {cfg}: differ {d}/{h}"
                    + ("  (good)" if d else "  <-- identical, lossy path NOT live!")
                )

    if args.verbose:
        print("\n-- per sentence --")
        for (g, i), per_cfg in sorted(sents.items()):
            md5s = {c: st.get("md5", "ERR") for c, st in per_cfg.items()}
            uniq = len(set(md5s.values()))
            print(f"  {g}#{i}: {uniq} distinct / {len(per_cfg)} configs")
            for c in sorted(per_cfg):
                st = per_cfg[c]
                if st.get("ok"):
                    print(
                        f"      {c:10s} dur={st['dur']:6.2f}s rms={st['rms']:.4f} "
                        f"md5={st['md5']}"
                    )
                else:
                    print(f"      {c:10s} UNREADABLE ({st.get('err', '?')})")

    healthy = s["bad"] == 0 and s["empty"] == 0
    applied = not (mc and ai == mc)
    print(
        f"\nVERDICT: files {'OK' if healthy else 'HAVE PROBLEMS'}; "
        f"compression {'applied' if applied else 'NOT applied (bug)'}"
    )


if __name__ == "__main__":
    main()
