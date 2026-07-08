"""Heatmap figure of the key×value(×rw) grid, Qwen vs VALL-E, shared scale.

Reads the tidy combined_summary.csv (from tools/combined_summary.py) and draws
one heatmap panel per model for a chosen metric, over the same axes as
tools/kv_grid.py --with-rw: rows K{bits}rw{rw}, cols V{bits}rw{rw}, plus an
fp16 baseline corner. Both panels share a single color scale so severity is
directly comparable across models. Cells are annotated; empty (non-existent)
configs are left blank.

  python tools/plot_kv_grid.py --summary results/combined_summary.csv \
      --metric cer --out results/kv_grid_cer.png

Sequential colormap by metric job: error metrics (cer/wer/collapse_pct) use a
dark=worse ramp; similarity metrics (cos_k/cos_v) use dark=higher. AR-only,
mean over datasets. Needs matplotlib (pip install --user matplotlib if absent).
"""

from __future__ import annotations

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_CFG_RE = re.compile(r"^K(\d+)V(\d+)@(\d+)$")
_LABELS = {
    "cer": "CER", "wer": "WER", "collapse_pct": "collapse %",
    "cos_k": "cos_k (key similarity)", "cos_v": "cos_v (value similarity)",
}
_ERROR_METRICS = {"cer", "wer", "collapse_pct"}


def _numkey(label: str) -> tuple:
    if label == "fp16":
        return (-999, 0)
    nums = [int(x) for x in re.findall(r"\d+", label)]
    return (-(nums[0] if nums else 0), nums[1] if len(nums) > 1 else 0)


def _prep(summary_path: str) -> pd.DataFrame:
    df = pd.read_csv(summary_path)
    df = df[df["group"] != "smoke"].copy()
    rows = []
    for _, r in df.iterrows():
        c = str(r["config"])
        if c.lower() == "fp16":
            rk, ck = "fp16", "fp16"
        else:
            m = _CFG_RE.match(c)
            if not m:  # nar/both — AR-only figure
                continue
            kb, vb, rw = m.group(1), m.group(2), m.group(3)
            rk, ck = f"K{kb}rw{rw}", f"V{vb}rw{rw}"
        rows.append({**r, "rowkey": rk, "colkey": ck})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", default="results/combined_summary.csv")
    ap.add_argument("--metric", default="cer", choices=list(_LABELS))
    ap.add_argument("--out", default="results/kv_grid.png")
    args = ap.parse_args()

    df = _prep(args.summary)
    metric = args.metric
    models = sorted(df["model"].unique())

    # Mean over datasets per (model, rowkey, colkey).
    cell = (
        df.groupby(["model", "rowkey", "colkey"])[metric].mean().reset_index()
    )
    rowkeys = sorted(cell["rowkey"].unique(), key=_numkey)
    colkeys = sorted(cell["colkey"].unique(), key=_numkey)

    # Shared color scale across both panels.
    vals = cell[metric].to_numpy(dtype=float)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        raise SystemExit(f"no data for metric {metric!r} (all blank?)")
    vmin, vmax = float(vals.min()), float(vals.max())
    cmap = plt.get_cmap("Reds" if metric in _ERROR_METRICS else "Blues")

    fig, axes = plt.subplots(
        1, len(models), figsize=(1.1 * len(colkeys) * len(models) + 2, 0.7 * len(rowkeys) + 2),
        squeeze=False,
    )
    for ax, model in zip(axes[0], models):
        grid = np.full((len(rowkeys), len(colkeys)), np.nan)
        sub = cell[cell["model"] == model]
        idx = {(r, c): v for r, c, v in zip(sub["rowkey"], sub["colkey"], sub[metric])}
        for i, rk in enumerate(rowkeys):
            for j, ck in enumerate(colkeys):
                if (rk, ck) in idx:
                    grid[i, j] = idx[(rk, ck)]
        im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(colkeys)), colkeys, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(rowkeys)), rowkeys, fontsize=8)
        ax.set_title(model, fontsize=12, fontweight="bold")
        span = (vmax - vmin) or 1.0
        for i in range(len(rowkeys)):
            for j in range(len(colkeys)):
                if not np.isnan(grid[i, j]):
                    dark = (grid[i, j] - vmin) / span > 0.6
                    ax.text(
                        j, i, f"{grid[i, j]:.3f}", ha="center", va="center",
                        fontsize=7, color="white" if dark else "black",
                    )
        # 2px surface gaps between cells (skill: separate adjacent fills).
        ax.set_xticks(np.arange(-0.5, len(colkeys), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(rowkeys), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="minor", length=0)

    fig.colorbar(im, ax=axes[0], shrink=0.8, label=_LABELS[metric])
    fig.suptitle(
        f"{_LABELS[metric]} by key/value bits × residual window "
        "(AR-only, mean over datasets)",
        fontsize=13,
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}  ({len(models)} panels, metric={metric})")


if __name__ == "__main__":
    main()
