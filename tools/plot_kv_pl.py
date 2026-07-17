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


def _grid(path: str, metric: str, rw=None):
    """(pivot key_bits×value_bits mean of metric, fp16 mean) from a score CSV.

    If ``rw`` is given, average only that residual window; else over all rw.
    fp16 has no rw so its mean is always over all fp16 rows.
    """
    d = pd.read_csv(path)
    d = d[d["group"] != "smoke"]
    fp16 = float(d[d["config"].astype(str).str.lower() == "fp16"][metric].mean())
    ar = d[d["config"].astype(str).str.match(_AR)].copy()
    ar["key_bits"] = ar["key_bits"].astype(int)
    ar["value_bits"] = ar["value_bits"].astype(int)
    ar["rw"] = ar["rw"].astype(int)
    if rw is not None:
        ar = ar[ar["rw"] == rw]
    piv = ar.pivot_table(index="key_bits", columns="value_bits",
                         values=metric, aggfunc="mean")
    return piv, fp16


def _draw(ax, piv, ks, vs, cmap, vmin, vmax, span, title):
    """One K×V heatmap panel; (0,0)=K4V4 top-left. Returns the image handle."""
    grid = np.full((len(ks), len(vs)), np.nan)
    for i, k in enumerate(ks):
        for j, v in enumerate(vs):
            if k in piv.index and v in piv.columns:
                grid[i, j] = piv.loc[k, v]
    im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(vs)), [f"V{v}" for v in vs])
    ax.set_yticks(range(len(ks)), [f"K{k}" for k in ks])
    ax.set_xlabel("value bits")
    ax.set_ylabel("key bits")
    ax.set_title(title, fontweight="bold", fontsize=10)
    for i in range(len(ks)):
        for j in range(len(vs)):
            if not np.isnan(grid[i, j]):
                dark = (grid[i, j] - vmin) / span > 0.6
                ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center",
                        fontsize=8, color="white" if dark else "black")
    ax.set_xticks(np.arange(-0.5, len(vs), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ks), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    return im


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", nargs="+", required=True,
                    metavar="label=path", help="one or more pl panels")
    ap.add_argument("--metric", default="cer", choices=list(_LABELS))
    ap.add_argument("--by-rw", action="store_true",
                    help="add rw as a dimension: rows=pl, columns=rw (each cell "
                    "a K×V heatmap). Default: average over rw.")
    ap.add_argument("--out", default="results/figures/kv_pl.png")
    args = ap.parse_args()

    srcs = [spec.split("=", 1) for spec in args.scores]  # [(label, path), ...]
    cmap = plt.get_cmap("Reds" if args.metric in _ERROR else "Blues")
    label = _LABELS[args.metric]

    # rw columns (only when --by-rw); union across files.
    rws = None
    if args.by_rw:
        rws = sorted(set().union(*[
            set(pd.read_csv(p)[pd.read_csv(p)["config"].astype(str).str.match(_AR)]
                ["rw"].astype(int)) for _, p in srcs
        ]))

    # Build every (pl, rw) pivot once; collect fp16 per pl and a shared scale.
    cells = {}          # (row_i, col_j) -> pivot
    fp16 = {}           # label -> fp16 mean
    for ri, (lab, path) in enumerate(srcs):
        cols = rws if args.by_rw else [None]
        for cj, rw in enumerate(cols):
            piv, f = _grid(path, args.metric, rw)
            cells[(ri, cj)] = piv
            fp16[lab] = f

    allv = np.concatenate([p.to_numpy(dtype=float).ravel() for p in cells.values()])
    allv = allv[~np.isnan(allv)]
    vmin, vmax = float(allv.min()), float(allv.max())
    span = (vmax - vmin) or 1.0
    ks = sorted(set().union(*[p.index for p in cells.values()]), reverse=True)
    vs = sorted(set().union(*[p.columns for p in cells.values()]), reverse=True)

    nrows, ncols = len(srcs), (len(rws) if args.by_rw else 1)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.1 * ncols + 1.5, 3.3 * nrows + 0.5),
                             squeeze=False)
    im = None
    for ri, (lab, _path) in enumerate(srcs):
        cols = rws if args.by_rw else [None]
        for cj, rw in enumerate(cols):
            title = f"{lab} rw{rw}" if args.by_rw else \
                    f"{lab}  (fp16 {label} {fp16[lab]:.3f})"
            im = _draw(axes[ri][cj], cells[(ri, cj)], ks, vs, cmap,
                       vmin, vmax, span, title)

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label=label)
    sup = label
    if args.by_rw:
        sup += "   (fp16: " + ", ".join(
            f"{lab} {fp16[lab]:.3f}" for lab, _ in srcs) + ")"
    fig.suptitle(sup, fontsize=14, fontweight="bold")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}  ({nrows}x{ncols} panels, metric={args.metric})")


if __name__ == "__main__":
    main()
