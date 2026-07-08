"""Cross-model summary: CER / WER / cos_k / cos_v per (model, dataset, config).

Joins each model's wav-score CSV (CER/WER, from tools/score_wav_dir.py) with
its attention-divergence CSV (cos_k/cos_v, from the divergence experiments)
into ONE tidy table, so Qwen and VALL-E sit side by side per eval group.

Each --model is ``label:wav_scores.csv:divergence.csv`` (the divergence path
may be omitted or '-' when there is no divergence CSV for that model):

  python tools/combined_summary.py \
      --model qwen:results/wav_scores.csv:results/kv_attn_100pg_noprotect.csv \
      --model valle:results/vallex_wav_scores.csv:results/vallex_attn_divergence.csv \
      --out results/combined_summary.csv

Notes / caveats baked in:
  * Divergence CSVs carry no ``config`` column — it is rebuilt from
    key_bits/value_bits/rw as ``K{kb}V{vb}@{rw}`` to match the wav-score labels.
  * cos_k/cos_v are averaged over UNPROTECTED layers only (protected layers sit
    at ~1.0 and would inflate the mean); n_layers is inferred per model as
    max(layer)+1 and protection from the protected_layers column.
  * WER is blank for any model whose wav-score CSV predates the WER column
    (e.g. a Qwen run scored before it existed — re-score to fill).
  * cos_k/cos_v are blank for configs with no divergence rows (fp16, and the
    VALL-E nar:/both: arms, which the AR-side recorder does not cover).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

COLLAPSE_CER = 0.5


_DIV_USECOLS = [
    "group", "layer", "key_bits", "value_bits", "rw",
    "protected_layers", "cos_k", "cos_v",
]


def _divergence_by_group_config(div_path: str) -> pd.DataFrame:
    """Mean cos_k/cos_v per (group, config), UNPROTECTED layers only.

    Streamed in chunks (the Qwen divergence CSV is ~20M rows) and fully
    vectorized — a row-wise apply on a file that size OOMs. cos columns may be
    blank (pos < rw → no compressed prefix); those read as NaN and are skipped
    by sum()/count().
    """
    # Pass 1: infer n_layers cheaply (one column).
    max_layer = 0
    for ch in pd.read_csv(div_path, usecols=["layer"], chunksize=5_000_000):
        max_layer = max(max_layer, int(ch["layer"].max()))
    n_layers = max_layer + 1

    # Pass 2: accumulate sum/count of cos_k/cos_v per (group, config).
    acc: dict = {}  # (group, config) -> [sum_k, cnt_k, sum_v, cnt_v]
    for ch in pd.read_csv(div_path, usecols=_DIV_USECOLS, chunksize=2_000_000):
        ch = ch[ch["group"] != "smoke"]
        pl = ch["protected_layers"].fillna(0).astype(int)
        ch = ch[~((ch["layer"] < pl) | (ch["layer"] >= (n_layers - pl)))]
        if ch.empty:
            continue
        cfg = (
            "K" + ch["key_bits"].astype(int).astype(str)
            + "V" + ch["value_bits"].astype(int).astype(str)
            + "@" + ch["rw"].astype(int).astype(str)
        )
        ch = ch.assign(config=cfg)
        g = ch.groupby(["group", "config"])
        sums = g[["cos_k", "cos_v"]].sum()
        cnts = g[["cos_k", "cos_v"]].count()
        for key in sums.index:
            a = acc.setdefault(key, [0.0, 0, 0.0, 0])
            a[0] += float(sums.loc[key, "cos_k"]); a[1] += int(cnts.loc[key, "cos_k"])
            a[2] += float(sums.loc[key, "cos_v"]); a[3] += int(cnts.loc[key, "cos_v"])

    rows = [
        {
            "group": grp, "config": cfg,
            "cos_k": sk / ck if ck else float("nan"),
            "cos_v": sv / cv if cv else float("nan"),
        }
        for (grp, cfg), (sk, ck, sv, cv) in acc.items()
    ]
    return pd.DataFrame(rows)


def _scores_by_group_config(wav_path: str) -> pd.DataFrame:
    d = pd.read_csv(wav_path)
    d = d[d["group"] != "smoke"].copy()
    if "wer" not in d.columns:
        d["wer"] = float("nan")
    g = d.groupby(["group", "config"])
    out = g.agg(
        n_wav=("cer", "size"),
        cer=("cer", "mean"),
        wer=("wer", "mean"),
        spk_sim=("spk_sim", "mean"),
    ).reset_index()
    out["collapse_pct"] = (
        g["cer"].apply(lambda s: (s > COLLAPSE_CER).mean()).values * 100
    )
    return out


def build(models: list[tuple[str, str, str]]) -> pd.DataFrame:
    frames = []
    for label, wav_path, div_path in models:
        scores = _scores_by_group_config(wav_path)
        scores.insert(0, "model", label)
        if div_path and div_path != "-" and os.path.exists(div_path):
            div = _divergence_by_group_config(div_path)
            scores = scores.merge(div, on=["group", "config"], how="left")
        else:
            scores["cos_k"] = float("nan")
            scores["cos_v"] = float("nan")
        frames.append(scores)
    df = pd.concat(frames, ignore_index=True)
    cols = [
        "model", "group", "config", "n_wav",
        "cer", "wer", "collapse_pct", "spk_sim", "cos_k", "cos_v",
    ]
    return df[cols].sort_values(["model", "group", "config"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="label:wav_csv:div_csv",
        help="Repeatable. Divergence CSV optional (use '-' to omit).",
    )
    ap.add_argument("--out", default="results/combined_summary.csv")
    ap.add_argument(
        "--markdown",
        default="",
        help="Also write metrics-only markdown tables (one per dataset) here.",
    )
    args = ap.parse_args()

    models = []
    for spec in args.model:
        parts = spec.split(":")
        if len(parts) == 2:
            label, wav, div = parts[0], parts[1], "-"
        elif len(parts) == 3:
            label, wav, div = parts
        else:
            raise SystemExit(f"bad --model {spec!r}; want label:wav_csv[:div_csv]")
        models.append((label, wav, div))

    df = build(models)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", None)
    show = df.copy()
    for c in ("cer", "wer", "spk_sim", "cos_k", "cos_v"):
        show[c] = show[c].round(4)
    show["collapse_pct"] = show["collapse_pct"].round(1)
    print(show.to_string(index=False))
    print(f"\nwrote {args.out}  ({len(df)} rows)")

    if args.markdown:
        _write_markdown(df, args.markdown)
        print(f"wrote {args.markdown}")


def _write_markdown(df: pd.DataFrame, path: str) -> None:
    """Metrics-only markdown: one table per dataset, no interpretation."""
    def cell(x):
        if pd.isna(x):
            return "—"
        return f"{x:.4f}" if abs(x) < 1 else f"{x:.1f}"

    cols = ["model", "config", "cer", "wer", "collapse_pct", "spk_sim",
            "cos_k", "cos_v"]
    head = "| model | config | CER | WER | collapse% | spkSim | cos_k | cos_v |"
    sep = "|---|---|---|---|---|---|---|---|"
    lines = ["# Metrics (CER / WER / collapse% / spkSim / cos_k / cos_v)", ""]
    for grp in sorted(df["group"].unique()):
        lines += [f"## {grp}", "", head, sep]
        sub = df[df["group"] == grp].sort_values(["model", "config"])
        for _, r in sub.iterrows():
            lines.append(
                "| " + " | ".join([
                    str(r["model"]), str(r["config"]),
                    cell(r["cer"]), cell(r["wer"]), cell(r["collapse_pct"]),
                    cell(r["spk_sim"]), cell(r["cos_k"]), cell(r["cos_v"]),
                ]) + " |"
            )
        lines.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
