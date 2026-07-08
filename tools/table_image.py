"""Render the combined metrics table as a PNG (for slides / sharing).

Reads combined_summary.csv (from tools/combined_summary.py) and draws a clean
table image: model, dataset, config, CER, WER, collapse%, spkSim, cos_k, cos_v.
Filter with --models / --groups / --configs to keep it slide-sized.

  python tools/table_image.py --summary results/combined_summary.csv \
      --out results/metrics_table.png
  # e.g. just VALL-E on librispeech_pc:
  python tools/table_image.py --models valle --groups librispeech_pc \
      --out results/metrics_valle_libri.png

Needs matplotlib (pip install --user matplotlib if absent).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

COLS = [
    ("model", "model"), ("group", "dataset"), ("config", "config"),
    ("cer", "CER"), ("wer", "WER"), ("collapse_pct", "collapse%"),
    ("spk_sim", "spkSim"), ("cos_k", "cos_k"), ("cos_v", "cos_v"),
]


def _fmt(col: str, v) -> str:
    if pd.isna(v):
        return "—"
    if col == "collapse_pct":
        return f"{v:.1f}"
    return f"{v:.4f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", default="results/combined_summary.csv")
    ap.add_argument("--models", default="", help="comma filter, e.g. valle")
    ap.add_argument("--groups", default="", help="comma filter, e.g. librispeech_pc")
    ap.add_argument("--configs", default="", help="comma filter, e.g. fp16,K4V4@64")
    ap.add_argument("--out", default="results/metrics_table.png")
    ap.add_argument("--title", default="KV-quant metrics")
    args = ap.parse_args()

    df = pd.read_csv(args.summary)
    df = df[df["group"] != "smoke"]
    for arg, col in (("models", "model"), ("groups", "group"), ("configs", "config")):
        val = getattr(args, arg)
        if val:
            keep = [x.strip() for x in val.split(",") if x.strip()]
            df = df[df[col].isin(keep)]
    df = df.sort_values(["model", "group", "config"]).reset_index(drop=True)
    if df.empty:
        raise SystemExit("no rows after filters")

    header = [label for _, label in COLS]
    text = [[_fmt(c, r[c]) for c, _ in COLS] for _, r in df.iterrows()]

    n = len(text)
    fig_h = 0.32 * (n + 1) + 0.6
    fig, ax = plt.subplots(figsize=(11, fig_h))
    ax.axis("off")
    tbl = ax.table(cellText=text, colLabels=header, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.3)

    ncol = len(header)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if row == 0:  # header
            cell.set_facecolor("#40466e")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#f4f5fa" if row % 2 else "#ffffff")
        if col in (0, 1):  # model / dataset columns emphasized
            cell.set_text_props(fontweight="bold")

    ax.set_title(args.title, fontsize=13, fontweight="bold", pad=12)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}  ({n} rows)")


if __name__ == "__main__":
    main()
