"""Quality-vs-compression trade-off curve — the thesis headline figure.

For each (key_bits, value_bits) config, computes the KV-cache compression ratio
vs fp16 and plots it against quality (CER / WER). Points on the lower-right
frontier (more compression, less error) are Pareto-optimal; that frontier and
its knee are the empirical "compression limit".

Compression math (matches turboquant_cache.memory_report, head_dim D):
  per token per tensor = idx_bytes + norm_bytes(=2, fp16 vector norm)
  REALIZED packing (as run): indices_per_byte = 8 // bits  -> idx = ceil(D / ipb)
      NOTE 8//bits is integer division, so 3-bit packs 2/byte just like 4-bit —
      i.e. K3 costs the SAME bytes as K4 in this implementation (no saving).
  IDEAL packing (--ideal): idx = ceil(D * bits / 8)  -> true sub-byte packing.
  ratio = fp16_bytes / (cbpt(key_bits) + cbpt(value_bits)),  fp16 = 2*D*2.
rw is asymptotically negligible (large context) and does not change the ratio,
so quality is averaged over rw per bit-pair -> one point per (K,V).

  python tools/plot_tradeoff.py \
      --scores pl2=results/grid_pl2_scores.csv pl0=results/grid_pl0_scores.csv \
      --metric cer --exclude-groups ellav_hard --out results/figures/tradeoff_cer.png
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

_AR = re.compile(r"^K(\d+)V(\d+)@(\d+)$")
_LABELS = {"cer": "CER", "wer": "WER"}


def _cbpt(bits: int, D: int, ideal: bool) -> int:
    """Compressed bytes per token per tensor (key or value)."""
    if ideal:
        idx = math.ceil(D * bits / 8)
    else:
        idx = math.ceil(D / (8 // bits))  # 8//bits: as-implemented packing
    return idx + 2  # + fp16 vector norm


def _ratio(kb: int, vb: int, D: int, ideal: bool) -> float:
    fp16 = 2 * D * 2  # K + V, fp16 (2 bytes/coord)
    return fp16 / (_cbpt(kb, D, ideal) + _cbpt(vb, D, ideal))


def _points(path: str, metric: str, exclude, D, ideal):
    d = pd.read_csv(path)
    d = d[d["group"] != "smoke"]
    if exclude:
        d = d[~d["group"].isin(exclude)]
    fp16 = float(d[d["config"].astype(str).str.lower() == "fp16"][metric].mean())
    ar = d[d["config"].astype(str).str.match(_AR)].copy()
    ar["key_bits"] = ar["key_bits"].astype(int)
    ar["value_bits"] = ar["value_bits"].astype(int)
    g = ar.groupby(["key_bits", "value_bits"])[metric].mean().reset_index()
    g["ratio"] = [_ratio(k, v, D, ideal) for k, v in zip(g["key_bits"], g["value_bits"])]
    g["label"] = "K" + g["key_bits"].astype(str) + "V" + g["value_bits"].astype(str)
    return g, fp16


def _pareto(g, metric):
    """Boolean mask: point is Pareto-optimal (no other has >=ratio and <=metric)."""
    mask = []
    for _, r in g.iterrows():
        dominated = (
            (g["ratio"] >= r["ratio"]) & (g[metric] <= r[metric])
            & ((g["ratio"] > r["ratio"]) | (g[metric] < r[metric]))
        ).any()
        mask.append(not dominated)
    return mask


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", nargs="+", required=True, metavar="label=path")
    ap.add_argument("--metric", default="cer", choices=list(_LABELS))
    ap.add_argument("--exclude-groups", default="ellav_hard")
    ap.add_argument("--head-dim", type=int, default=64, help="VALL-E head_dim=64")
    ap.add_argument("--ideal", action="store_true",
                    help="ideal sub-byte packing instead of the 8//bits as-run cost")
    ap.add_argument("--out", default="results/figures/tradeoff.png")
    args = ap.parse_args()

    excl = [g.strip() for g in args.exclude_groups.split(",") if g.strip()]
    metric, label = args.metric, _LABELS[args.metric]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    colors = plt.cm.tab10.colors

    for si, spec in enumerate(args.scores):
        name, path = spec.split("=", 1)
        g, fp16 = _points(path, metric, excl, args.head_dim, args.ideal)
        col = colors[si]
        ax.scatter(g["ratio"], g[metric], color=col, s=55, zorder=3, label=name)
        for _, r in g.iterrows():
            ax.annotate(r["label"], (r["ratio"], r[metric]),
                        textcoords="offset points", xytext=(5, 4), fontsize=7,
                        color=col)
        # Pareto frontier line (sorted by ratio)
        pm = _pareto(g, metric)
        pf = g[pd.Series(pm, index=g.index)].sort_values("ratio")
        ax.plot(pf["ratio"], pf[metric], color=col, lw=1.5, alpha=0.5, zorder=2)
        # fp16 baseline reference (ratio 1)
        ax.axhline(fp16, color=col, ls=":", lw=1, alpha=0.6)
        ax.annotate(f"{name} fp16 ({fp16:.3f})", (1.0, fp16),
                    textcoords="offset points", xytext=(4, 3), fontsize=7, color=col)

    ax.set_xlabel("KV-cache compression ratio  (× vs fp16)")
    ax.set_ylabel(label)
    pack = "ideal sub-byte packing" if args.ideal else "as-implemented (8//bits) packing"
    ax.set_title(f"{label} vs KV-cache compression\n"
                 f"lower-right is better; line = Pareto frontier  [{pack}]",
                 fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(title="protection", fontsize=9)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
