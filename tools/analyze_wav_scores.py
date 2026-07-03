"""Detailed breakdown of results/wav_scores.csv (from tools/score_wav_dir.py).

Prints, per config and per group x config:
  meanCER / medCER   over all sentences
  collapse%          share with CER > 0.5 (intelligibility failures)
  cleanCER           mean CER over NON-collapsed sentences only — quality when
                     it works, separated from how often it breaks
  spkSim             mean WavLM cosine vs the same sentence's fp16 wav
  durX / runaway%    median duration ratio vs fp16, and share with ratio > 2
                     (generation runaway / babble)

Plus: the worst sentences per config, and how collapse concentrates — sentences
broken under EVERY compressed config (hard content) vs under just one config
(config-specific damage).

CPU-only, seconds:  python tools/analyze_wav_scores.py --scores results/wav_scores.csv
"""

from __future__ import annotations

import argparse

import pandas as pd

COLLAPSE_CER = 0.5
RUNAWAY_RATIO = 2.0
BASELINE = "fp16"


def add_baseline_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Join each row with its sentence's fp16 reference: cer_fp16, dur_ratio."""
    base = (
        df[df["config"] == BASELINE]
        .set_index(["group", "idx"])[["cer", "dur_s"]]
        .rename(columns={"cer": "cer_fp16", "dur_s": "dur_fp16"})
    )
    out = df.join(base, on=["group", "idx"])
    out["dur_ratio"] = out["dur_s"] / out["dur_fp16"]
    return out


def summary(df: pd.DataFrame, by) -> pd.DataFrame:
    g = df.groupby(by)
    t = pd.DataFrame(
        {
            "n": g.size(),
            "meanCER": g["cer"].mean(),
            "medCER": g["cer"].median(),
            "collapse%": g["cer"].agg(lambda s: (s > COLLAPSE_CER).mean() * 100),
            "spkSim": g["spk_sim"].mean(),
        }
    )
    t["cleanCER"] = df[df["cer"] <= COLLAPSE_CER].groupby(by)["cer"].mean()
    if "dur_ratio" in df.columns:
        t["durX"] = g["dur_ratio"].median()
        t["runaway%"] = g["dur_ratio"].agg(
            lambda s: (s > RUNAWAY_RATIO).mean() * 100
        )
    return t


def collapse_concentration(df: pd.DataFrame) -> pd.Series:
    """#configs (out of the compressed ones) under which each sentence collapses
    -> how many sentences. Index 0 = robust everywhere; max = broken content."""
    comp = df[df["config"] != BASELINE]
    per_sentence = (
        comp.assign(c=comp["cer"] > COLLAPSE_CER)
        .groupby(["group", "idx"])["c"]
        .sum()
        .astype(int)
    )
    return per_sentence.value_counts().sort_index()


def worst_rows(df: pd.DataFrame, config: str, n: int) -> pd.DataFrame:
    cols = ["group", "idx", "cer", "spk_sim", "dur_ratio", "transcript"]
    cols = [c for c in cols if c in df.columns]
    w = df[df["config"] == config].nlargest(n, "cer")[cols].copy()
    if "transcript" in w.columns:
        w["transcript"] = w["transcript"].fillna("").str.slice(0, 60)
    return w


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", default="results/wav_scores.csv")
    parser.add_argument("--worst", type=int, default=8, help="Worst sentences per config.")
    args = parser.parse_args()

    df = pd.read_csv(args.scores)
    df = add_baseline_cols(df)
    fmt = lambda x: f"{x:.3f}"  # noqa: E731

    print(f"== overall by config ({df['group'].nunique()} groups) ==")
    print(summary(df, "config").to_string(float_format=fmt))

    print("\n== by group x config ==")
    print(summary(df, ["group", "config"]).to_string(float_format=fmt))

    print("\n== collapse concentration (how many compressed configs break each sentence) ==")
    conc = collapse_concentration(df)
    n_comp = df.loc[df["config"] != BASELINE, "config"].nunique()
    for k, v in conc.items():
        tag = " (robust)" if k == 0 else " (broken content)" if k == n_comp else ""
        print(f"  collapsed under {k}/{n_comp} configs: {v:4d} sentences{tag}")

    for cfg in sorted(df.loc[df["config"] != BASELINE, "config"].unique()):
        print(f"\n== worst {args.worst} sentences: {cfg} ==")
        print(worst_rows(df, cfg, args.worst).to_string(index=False, float_format=fmt))


if __name__ == "__main__":
    main()
