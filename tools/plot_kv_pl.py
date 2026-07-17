"""Key×Value heatmap per protection level (pl), from grid score CSVs.

x-axis = key bits, y-axis = value bits; one panel per pl, sharing a single
color scale so pl2 vs pl0 is directly comparable. Each cell is the chosen
metric averaged over residual window AND datasets (AR configs only; fp16 is
not a K/V cell, its value is shown in the title as a reference).

  python tools/plot_kv_pl.py \
      --scores pl2=results/grid_pl2_scores.csv pl0=results/grid_pl0_scores.csv \
      --metric cer --out results/figures/kv_pl_cer.png

Needs matplotlib (pip install --user matplotlib if absent).
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

_AR = re.compile(r"^K(\d+)V(\d+)@(\d+)$")  # AR only; excludes fp16 / -nar / -both
_LABELS = {"cer": "CER", "wer": "WER", "spk_sim": "speaker sim"}
_ERROR = {"cer", "wer"}


def _grid(path: str, metric: str):
    """(pivot value_bits×key_bits mean of metric, fp16 mean) from a score CSV."""
    d = pd.read_csv(path)
    d = d[d["group"] != "smoke"]
    fp16 = float(d[d["config"].astype(str).str.lower() == "fp16"][metric].mean())
    ar = d[d["config"].astype(str).str.match(_AR)].copy()
    ar["key_bits"] = ar["key_bits"].astype(int)
    ar["value_bits"] = ar["value_bits"].astype(int)
    # rows = key bits, columns = value bits (ordering set in main so that
    # (0,0) = K4V4, (0,1) = K4V3, ...).
    piv = ar.pivot_table(index="key_bits", columns="value_bits",
                         values=metric, aggfunc="mean")
    return piv, fp16


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", nargs="+", required=True,
                    metavar="label=path", help="one or more pl panels")
    ap.add_argument("--metric", default="cer", choices=list(_LABELS))
    ap.add_argument("--out", default="results/figures/kv_pl.png")
    args = ap.parse_args()

    panels = []
    for spec in args.scores:
        label, path = spec.split("=", 1)
        piv, fp16 = _grid(path, args.metric)
        panels.append((label, piv, fp16))

    # shared color scale across panels
    allv = np.concatenate([p.to_numpy(dtype=float).ravel() for _, p, _ in panels])
    allv = allv[~np.isnan(allv)]
    vmin, vmax = float(allv.min()), float(allv.max())
    span = (vmax - vmin) or 1.0
    cmap = plt.get_cmap("Reds" if args.metric in _ERROR else "Blues")
    label = _LABELS[args.metric]

    # Rows = key bits descending (K4 top → K2 bottom); columns = value bits
    # descending (V4 left → V2 right). So (0,0)=K4V4, (0,1)=K4V3, ...
    ks = sorted(set().union(*[p.index for _, p, _ in panels]), reverse=True)
    vs = sorted(set().union(*[p.columns for _, p, _ in panels]), reverse=True)

    im = None
    fig, axes = plt.subplots(1, len(panels),
                             figsize=(3.2 * len(panels) + 1.5, 3.6), squeeze=False)
    for ax, (name, piv, fp16) in zip(axes[0], panels):
        grid = np.full((len(ks), len(vs)), np.nan)
        for i, k in enumerate(ks):
            for j, v in enumerate(vs):
                if k in piv.index and v in piv.columns:
                    grid[i, j] = piv.loc[k, v]
        # default origin='upper' → row 0 (K4) at top → (0,0)=K4V4 top-left
        im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(vs)), [f"V{v}" for v in vs])
        ax.set_yticks(range(len(ks)), [f"K{k}" for k in ks])
        ax.set_xlabel("value bits")
        ax.set_ylabel("key bits")
        ax.set_title(f"{name}  (fp16 {label} {fp16:.3f})", fontweight="bold")
        for i in range(len(ks)):
            for j in range(len(vs)):
                if not np.isnan(grid[i, j]):
                    dark = (grid[i, j] - vmin) / span > 0.6
                    ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center",
                            fontsize=9, color="white" if dark else "black")
        ax.set_xticks(np.arange(-0.5, len(vs), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(ks), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="minor", length=0)

    fig.colorbar(im, ax=axes[0], shrink=0.8, label=label)
    fig.suptitle(label, fontsize=14, fontweight="bold")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}  ({len(panels)} panels, metric={args.metric})")


if __name__ == "__main__":
    main()
