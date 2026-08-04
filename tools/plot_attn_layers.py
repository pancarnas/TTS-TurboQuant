"""Per-layer KV-quant divergence profile for ONE model, across ALL its metrics.

Reads a layer-summary CSV (per (config, layer) mean of the divergence metrics) and
draws a small-multiple panel per metric: x = decoder layer, one line per config in
the compression ladder (K4V4@0, K3V3@0, K2V2@0). Separate figure per model, since
VALL-E (MHA, 12 layers, 9 metrics) and Qwen (GQA, 28 layers, 3 metrics) differ.

  python tools/plot_attn_layers.py --summary results/clone/vallex_layer_summary_mos.csv \
      --model "VALL-E-X (MHA, 12 layers)" --out results/final_39speaker/figures/attn_layers_valle.png
  python tools/plot_attn_layers.py --summary results/qwen_layer_summary.csv \
      --model "Qwen3-TTS (GQA, 28 layers)" --out results/final_39speaker/figures/attn_layers_qwen.png

Categorical palette validated CVD-safe (Okabe-Ito trio); identity is carried by a
legend + distinct markers, never colour alone.
"""

from __future__ import annotations

import argparse
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# colour + marker FOLLOW THE CONFIG (entity), fixed order = compression ladder
LADDER = ["K4V4@0", "K3V3@0", "K2V2@0"]
STYLE = {
    "K4V4@0": ("#0072B2", "o"),
    "K3V3@0": ("#E69F00", "s"),
    "K2V2@0": ("#D55E00", "^"),
}
# friendly labels; the rest fall back to the raw column name
NICE = {
    "attn_js": "attention JS divergence  (↓)",
    "attn_tv": "attention TV distance  (↓)",
    "attn_top1": "attention top-1 agreement  (↑)",
    "attn_dentropy": "Δ attention entropy",
    "out_cos": "output cosine  (↑)",
    "cos_k": "key cosine  (↑)",
    "cos_v": "value cosine  (↑)",
    "relmse_k": "key rel. MSE  (↓)",
    "relmse_v": "value rel. MSE  (↓)",
}
# preferred display order: divergences, entropy, then similarities
ORDER = ["attn_js", "attn_tv", "relmse_k", "relmse_v", "attn_dentropy",
         "attn_top1", "out_cos", "cos_k", "cos_v"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--model", required=True, help="title label for the model")
    ap.add_argument("--configs", default=",".join(LADDER),
                    help="compression-ladder configs to draw as lines")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = pd.read_csv(args.summary)
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    metrics = [m for m in ORDER if m in d.columns]
    metrics += [c for c in d.columns if c not in ("config", "layer") and c not in metrics]
    if not metrics:
        raise SystemExit("no metric columns found")

    ncol = min(3, len(metrics))
    nrow = math.ceil(len(metrics) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.2 * nrow),
                             squeeze=False)
    for k, metric in enumerate(metrics):
        ax = axes[k // ncol][k % ncol]
        for cfg in configs:
            sub = d[d["config"] == cfg].sort_values("layer")
            if sub.empty:
                continue
            col, mark = STYLE.get(cfg, ("#666666", "o"))
            ax.plot(sub["layer"].astype(int), sub[metric], color=col, lw=2,
                    marker=mark, ms=5, label=cfg)
        ax.set_title(NICE.get(metric, metric), fontsize=10)
        ax.set_xlabel("decoder layer", fontsize=8)
        ax.grid(True, lw=0.4, alpha=0.4)
        ax.tick_params(labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    # blank any unused panels
    for k in range(len(metrics), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=10,
               title="KV-quant config", ncol=len(labels))
    fig.suptitle(f"{args.model} — per-layer KV-quant divergence vs fp16",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}  ({len(metrics)} metrics x {nrow}x{ncol} panels, "
          f"configs={configs})")


if __name__ == "__main__":
    main()
