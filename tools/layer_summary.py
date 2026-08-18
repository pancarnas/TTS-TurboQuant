"""Reduce a raw per-(layer, pos) divergence CSV to a per-(config, layer) summary.

The generation-time divergence CSVs (VALL-E `--record-divergence` shards, Qwen
`kv_attn_divergence_experiment.py --out`) are per (group, idx, layer, pos,
config) and can reach 100+ MB; the layer figures only need the (config, layer)
means. Reads in chunks (pandas only, no torch), synthesizes the
``K<k>V<v>@<rw>`` config label from key_bits/value_bits/rw, and writes the
small CSV that `tools/plot_layer_profile.py` consumes.

  python tools/layer_summary.py \
      --divergence results/grid_cl_pl0_div_shard0.csv results/grid_cl_pl0_div_shard1.csv \
      --out results/vallex_layer_summary_cl.csv
"""

from __future__ import annotations

import argparse

import pandas as pd

METRICS = [
    "attn_js", "attn_tv", "attn_top1", "attn_dentropy",
    "out_cos", "cos_k", "cos_v", "relmse_k", "relmse_v",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--divergence", nargs="+", required=True,
                    help="One or more raw divergence CSVs (shards are fine).")
    ap.add_argument("--chunksize", type=int, default=2_000_000)
    ap.add_argument("--out", default="results/layer_summary.csv")
    args = ap.parse_args()

    sums: pd.DataFrame | None = None
    counts: pd.DataFrame | None = None
    for path in args.divergence:
        for chunk in pd.read_csv(path, chunksize=args.chunksize):
            metrics = [m for m in METRICS if m in chunk.columns]
            if "config" not in chunk.columns:
                chunk["config"] = (
                    "K" + chunk["key_bits"].astype("Int64").astype(str)
                    + "V" + chunk["value_bits"].astype("Int64").astype(str)
                    + "@" + chunk["rw"].astype("Int64").astype(str)
                )
            chunk = chunk.dropna(subset=["config"])
            grp = chunk.groupby(["config", "layer"])[metrics]
            s, c = grp.sum(), grp.count()
            sums = s if sums is None else sums.add(s, fill_value=0)
            counts = c if counts is None else counts.add(c, fill_value=0)

    if sums is None:
        raise SystemExit("no rows read")
    out = (sums / counts).reset_index()
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {out['config'].nunique()} configs x "
          f"{out['layer'].nunique()} layers")


if __name__ == "__main__":
    main()
