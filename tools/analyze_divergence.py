"""Aggregate the kv_attn divergence CSV: by layer, by position, and vs collapse.

Reads the (potentially multi-GB) per-(layer, pos, config, sentence) CSV from
kv_attn_divergence_experiment.py in chunks and prints:

  1. mean divergence by config          — same as the job-log summary
  2. attn_js / cos_k by layer x config  — which layers hurt most; decides whether
                                          first/last-N protection targets them
  3. attn_js by position bucket         — does error grow with sequence length
                                          (informs the residual window choice)
  4. collapsed vs clean sentences       — with --scores, joins wav_scores.csv and
                                          compares divergence on sentences that
                                          collapsed (CER > 0.5) vs survived

CPU-only:
  python tools/analyze_divergence.py \
      --divergence results/kv_attn_100pg_noprotect.csv \
      --scores results/wav_scores.csv
"""

from __future__ import annotations

import argparse

import pandas as pd

COLLAPSE_CER = 0.5
METRICS = ["attn_js", "attn_top1", "out_cos", "cos_k", "cos_v", "relmse_k", "relmse_v"]


def config_col(df: pd.DataFrame) -> pd.Series:
    return (
        "K"
        + df["key_bits"].astype(int).astype(str)
        + "V"
        + df["value_bits"].astype(int).astype(str)
        + "@"
        + df["rw"].astype(int).astype(str)
    )


class GroupMean:
    """Streaming weighted mean of METRICS over an arbitrary key set."""

    def __init__(self, keys: list[str]):
        self.keys = keys
        self._parts: list[pd.DataFrame] = []

    def add(self, chunk: pd.DataFrame) -> None:
        g = chunk.groupby(self.keys, observed=True)[METRICS].sum()
        g["_n"] = chunk.groupby(self.keys, observed=True).size()
        self._parts.append(g)

    def result(self) -> pd.DataFrame:
        total = pd.concat(self._parts).groupby(level=self.keys, observed=True).sum()
        return total[METRICS].div(total["_n"], axis=0)


def load_collapse_map(scores_path: str) -> pd.DataFrame:
    """(group, idx, config) -> collapsed flag from wav_scores.csv."""
    s = pd.read_csv(scores_path)
    s = s[s["config"] != "fp16"]
    s["collapsed"] = s["cer"] > COLLAPSE_CER
    return s[["group", "idx", "config", "collapsed"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--divergence", default="results/kv_attn_100pg_noprotect.csv")
    parser.add_argument("--scores", default=None, help="wav_scores.csv for the collapse join.")
    parser.add_argument("--pos-bin", type=int, default=200)
    parser.add_argument("--chunksize", type=int, default=2_000_000)
    args = parser.parse_args()

    by_config = GroupMean(["config"])
    by_layer = GroupMean(["config", "layer"])
    by_pos = GroupMean(["config", "posbin"])
    by_collapse = GroupMean(["config", "collapsed"]) if args.scores else None
    collapse_map = load_collapse_map(args.scores) if args.scores else None

    rows = 0
    for chunk in pd.read_csv(args.divergence, chunksize=args.chunksize):
        chunk = chunk.dropna(subset=["attn_js"])
        chunk["config"] = config_col(chunk)
        chunk["posbin"] = (chunk["pos"] // args.pos_bin) * args.pos_bin
        by_config.add(chunk)
        by_layer.add(chunk)
        by_pos.add(chunk)
        if by_collapse is not None:
            merged = chunk.merge(collapse_map, on=["group", "idx", "config"], how="inner")
            if len(merged):
                by_collapse.add(merged)
        rows += len(chunk)
        print(f"\r  scanned {rows:,} rows", end="", flush=True)
    print()

    fmt = lambda x: f"{x:.4f}"  # noqa: E731

    print("\n== mean divergence by config ==")
    print(by_config.result().to_string(float_format=fmt))

    layer = by_layer.result().reset_index()
    for metric in ("attn_js", "cos_k"):
        print(f"\n== {metric} by layer x config ==")
        print(
            layer.pivot(index="layer", columns="config", values=metric).to_string(
                float_format=fmt
            )
        )

    pos = by_pos.result().reset_index()
    print(f"\n== attn_js by position bucket (bin={args.pos_bin}) x config ==")
    print(
        pos.pivot(index="posbin", columns="config", values="attn_js").to_string(
            float_format=fmt
        )
    )

    if by_collapse is not None:
        print("\n== divergence on collapsed vs clean sentences ==")
        print(
            by_collapse.result()[["attn_js", "attn_top1", "out_cos", "cos_k"]].to_string(
                float_format=fmt
            )
        )


if __name__ == "__main__":
    main()
