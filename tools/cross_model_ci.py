"""Matched cross-model table with seed error bars + Wilson collapse CIs.

Self-contained (pandas + stdlib only — deliberately does NOT import the
turboquant/torch chain, so it runs on a laptop). For the configs both seed runs
share, reports per model:
  * WER / CER mean +/- std ACROSS SEEDS (per-seed mean over sentences first),
  * collapse rate (CER > 0.5) with a 95% Wilson score interval.

This closes the two rigor items in one artifact: Qwen multi-seed error bars, and
a confidence interval on every collapse rate (Qwen n=300, VALL-E n=300).

  python tools/cross_model_ci.py --group librispeech_pc \
      --qwen results/qwen_seed_scores.csv --valle results/seed_cl_pl0_scores.csv \
      --out results/cross_model_ci.md
"""

from __future__ import annotations

import argparse
import math
import os

import pandas as pd

COLLAPSE_CER = 0.5
Z = 1.96
CONFIGS = ["fp16", "K4V4@0", "K4V2@0", "K3V3@0", "K2V2@0"]  # shared by both seed runs


def wilson(k: int, n: int, z: float = Z) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, center - half), min(1.0, center + half)


def rows_for(path: str, group: str, model: str) -> list[dict]:
    d = pd.read_csv(path)
    d = d[d["group"] == group]
    out = []
    for cfg in CONFIGS:
        g = d[d["config"] == cfg]
        if g.empty:
            continue
        seed_means_w = g.groupby("seed")["wer"].mean()
        seed_means_c = g.groupby("seed")["cer"].mean()
        k = int((g["cer"] > COLLAPSE_CER).sum())
        n = len(g)
        lo, hi = wilson(k, n)
        out.append({
            "model": model, "config": cfg, "n": n,
            "n_seeds": g["seed"].nunique(),
            "wer": seed_means_w.mean(),
            "wer_sd": seed_means_w.std(ddof=1) if len(seed_means_w) > 1 else 0.0,
            "cer": seed_means_c.mean(),
            "cer_sd": seed_means_c.std(ddof=1) if len(seed_means_c) > 1 else 0.0,
            "collapse": 100 * k / n, "coll_lo": 100 * lo, "coll_hi": 100 * hi,
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--group", default="librispeech_pc")
    ap.add_argument("--qwen", default="results/qwen_seed_scores.csv")
    ap.add_argument("--valle", default="results/seed_cl_pl0_scores.csv")
    ap.add_argument("--out", default="results/cross_model_ci.md")
    args = ap.parse_args()

    rows = (rows_for(args.qwen, args.group, "Qwen (GQA, 12.5 Hz)")
            + rows_for(args.valle, args.group, "VALL-E (MHA, 75 Hz)"))
    df = pd.DataFrame(rows)

    lines = [f"# Cross-model, seed error bars + Wilson collapse CIs "
             f"({args.group})", "",
             "WER/CER = mean ± std across seeds; collapse = CER>0.5 with 95% "
             "Wilson interval.", "",
             "| model | config | n | WER | CER | collapse% [95% CI] |",
             "|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['model']} | {r['config']} | {r['n']} "
            f"| {r['wer']:.3f} ± {r['wer_sd']:.3f} "
            f"| {r['cer']:.3f} ± {r['cer_sd']:.3f} "
            f"| {r['collapse']:.0f}% [{r['coll_lo']:.0f}, {r['coll_hi']:.0f}] |")
    md = "\n".join(lines)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(md + "\n")
    print(md)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
