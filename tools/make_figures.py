"""Generate the full KV-quant figure set into results/figures/.

Per metric (CER, WER, collapse%, cos_k, cos_v): a Qwen-vs-VALL-E heatmap
comparison on the literal K{bits}rw{rw} × V{bits}rw{rw} grid —

  * averaged/<metric>.png   — mean over the 3 datasets both models share
                              (seedtts_en, librispeech_pc, ellav_hard);
                              libritts_long is excluded here (Qwen never ran it)
                              and shown only in its own per-dataset map.
  * by_dataset/<metric>_<dataset>.png — one per dataset; both models always get
                              a panel, blank ("(none)") where a model has no
                              data (Qwen WER, Qwen libritts_long, VALL-E cos_*
                              outside librispeech_pc).

Both panels share one color scale so severity is directly comparable.

  python tools/make_figures.py --summary results/combined_summary.csv \
      --outdir results/figures/heatmaps
"""

from __future__ import annotations

import argparse
import os
import sys

_TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _TOOLS)
sys.path.insert(0, os.path.dirname(_TOOLS))

import plot_kv_grid as P  # noqa: E402

METRICS = [
    ("cer", "CER"), ("wer", "WER"), ("collapse_pct", "collapse%"),
    ("cos_k", "cos_k (key similarity)"), ("cos_v", "cos_v (value similarity)"),
]
COMMON = ["seedtts_en", "librispeech_pc", "ellav_hard"]
ALL_DATASETS = ["seedtts_en", "librispeech_pc", "ellav_hard", "libritts_long"]
MODELS = ["qwen", "valle"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", default="results/combined_summary.csv")
    ap.add_argument("--outdir", default="results/figures/heatmaps")
    args = ap.parse_args()

    df = P._prep(args.summary)
    n = 0
    for key, label in METRICS:
        # 1) headline: averaged over the 3 shared datasets
        avg = df[df["group"].isin(COMMON)]
        if P.render(
            avg, key, os.path.join(args.outdir, "averaged", f"{key}.png"),
            models=MODELS,
            title=f"{label} — Qwen vs VALL-E, mean over {len(COMMON)} shared datasets",
        ):
            n += 1
        # 2) per-dataset detail
        for ds in ALL_DATASETS:
            sub = df[df["group"] == ds]
            if P.render(
                sub, key,
                os.path.join(args.outdir, "by_dataset", f"{key}_{ds}.png"),
                models=MODELS, title=f"{label} — Qwen vs VALL-E, {ds}",
            ):
                n += 1
    print(f"\n{n} figures written under {args.outdir}/")


if __name__ == "__main__":
    main()
