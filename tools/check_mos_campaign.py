"""Sanity-check the 39-speaker MOS generation/scoring campaign (read-only).

Run on the box where the wavs live. Reports, in four sections:
  1. FILES        — wav count, unparseable/foreign files, distinct idxs & configs.
  2. COMPLETENESS — for the idxs in --idx-file, which of the expected configs are
                    missing (so you catch a shard that died mid-run).
  3. DURATION     — min/mean/max seconds + count of suspiciously short (<1 s) clips
                    (empty / collapsed generations).
  4. SPEAKERS     — distinct reference speakers among the generated idxs + per-speaker
                    sentence count (needs iter_eval_items; skipped if torch absent).
  5. METRICS      — only with --scores: per-config mean CER/WER/spk_sim + collapse%
                    (frac CER>0.5) and an fp16-baseline WER sanity check.

  python tools/check_mos_campaign.py \
      --audio-dir models/VALL-E-X/benchmarks/outputs/grid_clone_mos \
      --idx-file mos_idx_list.txt \
      --scores results/clone/grid_clone_mos_scores.csv     # optional, after scoring
"""

from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict

CONFIGS_28 = (
    "fp16,K2V2@0,K2V2@128,K2V2@64,K2V3@0,K2V3@128,K2V3@64,K2V4@0,K2V4@128,K2V4@64,"
    "K3V2@0,K3V2@128,K3V2@64,K3V3@0,K3V3@128,K3V3@64,K3V4@0,K3V4@128,K3V4@64,"
    "K4V2@0,K4V2@128,K4V2@64,K4V3@0,K4V3@128,K4V3@64,K4V4@0,K4V4@128,K4V4@64"
)
_WAV_RE = re.compile(
    r"^vallex_(?P<group>.+)_(?P<idx>\d+)_(?P<arm>[a-z]+)_s(?P<seed>\d+)"
    r"(?:_t[0-9.]+)?_(?P<config>.+)\.wav$"
)


def _read_idx_file(path: str) -> set[int]:
    idxs = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                idxs.add(int(line))
    return idxs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audio-dir", default=None,
                    help="dir of generated wavs; omit for a metrics-only check "
                         "(then only --scores is used, e.g. locally)")
    ap.add_argument("--idx-file", default=None,
                    help="expected idxs; omit to use whatever idxs are present")
    ap.add_argument("--configs", default=CONFIGS_28,
                    help="comma list of configs each idx should have")
    ap.add_argument("--group", default="librispeech_pc")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--scores", default=None, help="scores CSV (optional; after scoring)")
    ap.add_argument("--short-sec", type=float, default=1.0)
    args = ap.parse_args()

    want_cfgs = [c.strip() for c in args.configs.split(",") if c.strip()]
    have_dir = bool(args.audio_dir) and os.path.isdir(args.audio_dir)

    # --- 1. FILES ---
    present: dict[int, set[str]] = defaultdict(set)
    paths: dict[tuple[int, str], str] = {}
    foreign = 0
    n_wav = 0
    print("=== 1. FILES ===")
    if have_dir:
        for fn in os.listdir(args.audio_dir):
            if not fn.endswith(".wav"):
                continue
            n_wav += 1
            m = _WAV_RE.match(fn)
            if not m or m["group"] != args.group:
                foreign += 1
                continue
            idx, cfg = int(m["idx"]), m["config"]
            present[idx].add(cfg)
            paths[(idx, cfg)] = os.path.join(args.audio_dir, fn)
        all_cfgs = sorted({c for cfgs in present.values() for c in cfgs})
        print(f"  wavs: {n_wav} ({foreign} foreign/unparseable)")
        print(f"  distinct idxs present: {len(present)}")
        print(f"  distinct configs present: {len(all_cfgs)}  (want {len(want_cfgs)})")
    else:
        print("  (no --audio-dir; skipping file/completeness/duration checks)")

    # expected idxs: explicit file, else whatever is present
    want_idxs = _read_idx_file(args.idx_file) if args.idx_file else set(present)

    # --- 2. COMPLETENESS ---
    if have_dir:
        print("=== 2. COMPLETENESS (idx x config) ===")
        missing_pairs = []
        complete_idxs = 0
        for idx in sorted(want_idxs):
            have = present.get(idx, set())
            miss = [c for c in want_cfgs if c not in have]
            if not miss:
                complete_idxs += 1
            else:
                missing_pairs += [(idx, c) for c in miss]
        idxs_absent = sorted(i for i in want_idxs if i not in present)
        print(f"  fully-complete idxs: {complete_idxs}/{len(want_idxs)}")
        print(f"  missing (idx,config) pairs: {len(missing_pairs)}")
        if idxs_absent:
            print(f"  idxs with ZERO wavs ({len(idxs_absent)}): {idxs_absent[:20]}")
        if missing_pairs:
            print(f"  examples: {missing_pairs[:10]}")

    # --- 3. DURATION ---
    if have_dir:
        print("=== 3. DURATION ===")
        try:
            import soundfile as sf
            durs, short = [], []
            for (idx, cfg), p in paths.items():
                try:
                    info = sf.info(p)
                    d = info.frames / info.samplerate
                except Exception:
                    d = 0.0
                durs.append(d)
                if d < args.short_sec:
                    short.append((idx, cfg, round(d, 2)))
            if durs:
                print(f"  n={len(durs)}  min={min(durs):.2f}s  "
                      f"mean={sum(durs)/len(durs):.2f}s  max={max(durs):.2f}s")
            print(f"  suspiciously short (<{args.short_sec}s): {len(short)}")
            if short:
                print(f"    examples: {short[:10]}")
        except ImportError:
            print("  (soundfile not importable — skipped)")

    # --- 4. SPEAKERS ---
    print("=== 4. SPEAKERS ===")
    try:
        from turboquant.eval_sentences import iter_eval_items
        items = iter_eval_items([args.group], None, args.data_dir)

        def spk(i):
            ref = (getattr(items[i], "ref_audio", None)
                   or getattr(items[i], "ground_truth_audio", None))
            return os.path.basename(ref).split("-")[0] if ref else str(i)

        # idx source: generated wavs if scanned, else the idxs in the scores CSV
        spk_idxs = set(present)
        if not spk_idxs and args.scores and os.path.exists(args.scores):
            import pandas as pd
            sc = pd.read_csv(args.scores)
            spk_idxs = {int(i) for i in sc[sc["group"] == args.group]["idx"].unique()}
        per_spk = defaultdict(int)
        for i in spk_idxs:
            if i < len(items):
                per_spk[spk(i)] += 1
        print(f"  distinct speakers among generated idxs: {len(per_spk)}")
        counts = sorted(per_spk.values())
        if counts:
            print(f"  sentences/speaker: min={counts[0]} max={counts[-1]} "
                  f"mean={sum(counts)/len(counts):.1f}")
        print("  per-speaker: " + ", ".join(f"{s}={n}" for s, n in sorted(per_spk.items())))
    except Exception as exc:
        print(f"  (skipped: {type(exc).__name__}: {exc})")

    # --- 5. METRICS ---
    if args.scores:
        print("=== 5. METRICS (from scores CSV) ===")
        try:
            import pandas as pd
            d = pd.read_csv(args.scores)
            d = d[d["group"] == args.group]
            print(f"  rows: {len(d)}  configs: {d['config'].nunique()}")
            fp = d[d["config"].astype(str).str.lower() == "fp16"]
            if len(fp):
                print(f"  fp16 baseline: CER={fp['cer'].mean():.3f} "
                      f"WER={fp['wer'].mean():.3f} (expect WER ~0.09)")
            print(f"  {'config':<12}{'n':>5}{'CER':>8}{'WER':>8}{'spkSim':>9}{'collapse%':>11}")
            for cfg in sorted(d["config"].astype(str).unique()):
                sub = d[d["config"].astype(str) == cfg]
                collapse = 100.0 * (sub["cer"] > 0.5).mean()
                ss = sub["spk_sim"]
                ss_m = ss.astype(float).mean() if ss.notna().any() else float("nan")
                print(f"  {cfg:<12}{len(sub):>5}{sub['cer'].mean():>8.3f}"
                      f"{sub['wer'].mean():>8.3f}{ss_m:>9.3f}{collapse:>10.1f}%")
        except Exception as exc:
            print(f"  (could not read scores: {type(exc).__name__}: {exc})")


if __name__ == "__main__":
    main()
