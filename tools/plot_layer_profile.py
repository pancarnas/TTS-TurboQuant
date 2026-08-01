"""Per-layer attention-divergence profile: VALL-E (MHA) vs Qwen (GQA).

At a shared config (default K4V4@0, where both models have ~identical KV
reconstruction, cos_k/cos_v ~0.995), plots mean attn_js per decoder layer for
each model on ONE shared linear axis. The point of the figure is the magnitude
+ shape contrast, so a single y-axis (never dual) is deliberate: Qwen's error is
an order of magnitude larger AND concentrates in a layer-5 hotspot, while
VALL-E's is small and flat — the mechanical reason MHA degrades gracefully where
GQA collapses.

Reads the small per-layer summaries (config, layer, attn_js, cos_k, cos_v):
  results/vallex_layer_summary_cl.csv   (VALL-E, 12 layers)
  results/qwen_layer_summary.csv        (Qwen, 28 layers)

  python tools/plot_layer_profile.py --config K4V4@0 \
      --out results/figures/layer_profile_attnjs.png
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

# dataviz reference palette (light mode) — categorical slots 1 & 2 (CVD-safe pair)
VALLE = "#2a78d6"   # blue  — slot 1
QWEN = "#eb6834"    # orange — slot 2
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def _profile(path: str, config: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    d = d[d["config"] == config].copy()
    d["layer"] = d["layer"].astype(int)
    return d.sort_values("layer")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--valle", default="results/vallex_layer_summary_cl.csv")
    ap.add_argument("--qwen", default="results/qwen_layer_summary.csv")
    ap.add_argument("--config", default="K4V4@0",
                    help="shared config both models carry (default K4V4@0)")
    ap.add_argument("--out", default="results/figures/layer_profile_attnjs.png")
    args = ap.parse_args()

    v = _profile(args.valle, args.config)
    q = _profile(args.qwen, args.config)
    if v.empty or q.empty:
        raise SystemExit(f"config {args.config} missing in one model "
                         f"(valle={len(v)}, qwen={len(q)} rows)")

    def spread(df):
        lo = max(df["attn_js"].min(), 1e-9)
        return df["attn_js"].max() / lo

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.plot(q["layer"], q["attn_js"], color=QWEN, lw=2, marker="o", ms=5,
            zorder=3)
    ax.plot(v["layer"], v["attn_js"], color=VALLE, lw=2, marker="o", ms=5,
            zorder=3)

    # Qwen layer-5 hotspot annotation
    qpk = q.loc[q["attn_js"].idxmax()]
    ax.annotate(f"hotspot: layer {int(qpk['layer'])}\nattn_js {qpk['attn_js']:.3f}",
                xy=(qpk["layer"], qpk["attn_js"]),
                xytext=(qpk["layer"] + 3.2, qpk["attn_js"] - 0.004),
                color=INK2, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2))

    # direct labels (identity carried by the colored line beside each)
    ax.text(q["layer"].iloc[-1] + 0.4, q["attn_js"].iloc[-1],
            f"Qwen (GQA, 28 layers)\nspread {spread(q):.1f}×",
            color=QWEN, fontsize=10, va="center", fontweight="bold")
    ax.text(13.0, 0.0045,
            f"VALL-E (MHA, 12 layers) — flat, spread {spread(v):.1f}×",
            color=VALLE, fontsize=10, va="center", fontweight="bold")

    ax.set_ylim(bottom=0)
    ax.set_xlim(-0.6, max(q["layer"].max(), v["layer"].max()) + 7)
    ax.set_xlabel("decoder layer", color=INK2, fontsize=10)
    ax.set_ylabel("attention divergence  (mean attn_js vs fp16)",
                  color=INK2, fontsize=10)
    ax.set_title(
        f"Per-layer attention divergence at {args.config}  "
        f"(identical KV fidelity, cos$_k$/cos$_v$ ≈ 0.995)",
        color=INK, fontsize=11.5, fontweight="bold", pad=12)

    ax.grid(True, axis="y", color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {args.out}")
    print(f"  VALL-E {args.config}: peak {v['attn_js'].max():.4f}  spread {spread(v):.1f}x  ({len(v)} layers)")
    print(f"  Qwen   {args.config}: peak {q['attn_js'].max():.4f} @layer "
          f"{int(qpk['layer'])}  spread {spread(q):.1f}x  ({len(q)} layers)")


if __name__ == "__main__":
    main()
