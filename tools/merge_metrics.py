"""Master per-config metrics table: audio + perplexity + attention distance.

Joins the three per-experiment CSVs into ONE per-config row so a reviewer sees
every metric together:
  * audio       (tools/score_wav_dir.py)  — CER, WER, collapse%, spk_sim
  * perplexity  (vallex_ppl_divergence.py --out)        — ppl, nll, delta_nll,
                                                           kl_mean, top1_agree,
                                                           first_flip (median)
  * attention   (vallex_ppl_divergence.py --divergence) — attn_js, attn_tv,
                distance                    attn_top1, out_cos, cos_k, cos_v,
                                            relmse_k, relmse_v (unprotected
                                            layers only)

PPL and attention distances exist only where teacher-forcing ran (librispeech_pc,
AR configs), so by default everything is restricted to that group for a
consistent same-sentences comparison. Audio-only metrics over more datasets are
better read from the score CSV directly.

  python tools/merge_metrics.py \
      --scores results/grid_pl2_scores.csv \
      --ppl    results/grid_ppl_pl2.csv \
      --divergence results/grid_div_pl2.csv \
      --group librispeech_pc --out results/master_pl2.csv
"""

from __future__ import annotations

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

_AR = re.compile(r"^K(\d+)V(\d+)@(\d+)$")
COLLAPSE_CER = 0.5


def _audio(path: str, group: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    d = d[d["group"] == group] if group else d[d["group"] != "smoke"]
    d = d[d["config"].astype(str).str.match(_AR)]
    g = d.groupby("config")
    out = g.agg(cer=("cer", "mean"), wer=("wer", "mean"),
                spk_sim=("spk_sim", "mean")).reset_index()
    out["collapse_pct"] = (
        g["cer"].apply(lambda s: (s > COLLAPSE_CER).mean()).values * 100
    )
    return out


def _ppl(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=["config"])
    d = pd.read_csv(path)
    d = d[d["config"].astype(str).str.match(_AR)].copy()
    d["first_flip"] = pd.to_numeric(d.get("first_flip"), errors="coerce")
    return d.groupby("config").agg(
        ppl=("ppl", "mean"), nll=("nll", "mean"), delta_nll=("delta_nll", "mean"),
        kl_mean=("kl_mean", "mean"), top1_agree=("top1_agree", "mean"),
        first_flip=("first_flip", "median"),
    ).reset_index()


def _divergence(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=["config"])
    d = pd.read_csv(path)
    n_layers = int(d["layer"].max()) + 1
    pl = d["protected_layers"].fillna(0).astype(int)
    d = d[~((d["layer"] < pl) | (d["layer"] >= (n_layers - pl)))]  # unprotected only
    d = d.copy()
    d["config"] = ("K" + d["key_bits"].astype(int).astype(str)
                   + "V" + d["value_bits"].astype(int).astype(str)
                   + "@" + d["rw"].astype(int).astype(str))
    metrics = [m for m in ("attn_js", "attn_tv", "attn_top1", "out_cos",
                           "cos_k", "cos_v", "relmse_k", "relmse_v")
               if m in d.columns]
    return d.groupby("config")[metrics].mean().reset_index()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--ppl", default="")
    ap.add_argument("--divergence", default="")
    ap.add_argument("--group", default="librispeech_pc",
                    help="restrict to one dataset for aligned metrics (default "
                    "librispeech_pc; '' = all non-smoke, but PPL/attn are libri-only)")
    ap.add_argument("--out", default="results/master_metrics.csv")
    args = ap.parse_args()

    m = _audio(args.scores, args.group)
    m = m.merge(_ppl(args.ppl), on="config", how="left")
    m = m.merge(_divergence(args.divergence), on="config", how="left")

    bits = m["config"].str.extract(_AR)
    m.insert(1, "key_bits", bits[0].astype(int))
    m.insert(2, "value_bits", bits[1].astype(int))
    m.insert(3, "rw", bits[2].astype(int))
    m = m.sort_values(["key_bits", "value_bits", "rw"],
                      ascending=[False, False, True]).reset_index(drop=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    m.to_csv(args.out, index=False)
    pd.set_option("display.width", 240)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    print(m.round(4).to_string(index=False))
    print(f"\nwrote {args.out}  ({len(m)} configs, group={args.group or 'all'})")


if __name__ == "__main__":
    main()
