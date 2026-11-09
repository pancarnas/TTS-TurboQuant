"""Summarise a 14_qwen_keyfix.sh run into one markdown file.

Per cell (plain / g32 / pre / g32pre) and config:
  * objective: CER, WER, collapse rate (CER > 0.5, as tools/cross_model_ci.py),
    spk_sim vs the cell's own fp16 wav — from results/qwen_keyfix_<cell>_scores.csv
  * divergence: mean attn_js, cos_k, cos_v — from results/qwen_keyfix_<cell>_seed*.csv
  * the per-layer attn_js profile for K2V2@0 (the layer-5 hotspot question)

The 'plain' baseline may come from 13_qwen_divergence.sh instead: pass
--plain-div results/qwen_div_seed0.csv --plain-scores results/qwen_seed_scores.csv.

  python tools/summarize_keyfix.py --results results --out results/keyfix_summary.md
"""

from __future__ import annotations

import argparse
import glob
import os

COLLAPSE_CER = 0.5
CELLS = ["plain", "g32", "pre", "g32pre"]


def _label(kb, vb, rw, kg, pre):
    return f"K{int(kb)}V{int(vb)}@{int(rw)}" + (f"+g{int(kg)}" if kg else "") + ("+pre" if pre else "")


def _md(df, floatfmt=".3f", index=False) -> str:
    """Minimal markdown table (no tabulate dependency on the GPU box)."""
    if index:
        df = df.reset_index()
    cols = list(df.columns)

    def fmt(v):
        if isinstance(v, float):
            return "" if v != v else format(v, floatfmt)  # NaN -> blank
        return str(v)

    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for i in range(len(df)):  # per-column access keeps ints as ints (iterrows upcasts)
        out.append("| " + " | ".join(fmt(df[c].iloc[i].item() if hasattr(df[c].iloc[i], "item") else df[c].iloc[i]) for c in cols) + " |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/keyfix_summary.md")
    ap.add_argument("--plain-div", default=None, help="divergence CSV to use as the 'plain' cell")
    ap.add_argument("--plain-scores", default=None, help="scores CSV to use as the 'plain' cell")
    ap.add_argument("--talker-min-pos", type=int, default=17,
                    help="divergence rows with pos < this are the code predictor (its sequence is "
                         "at most num_code_groups=16 long) and are dropped; 0 keeps every row")
    args = ap.parse_args()

    import pandas as pd

    lines = ["# Key-fix experiment summary (Qwen3-TTS)", ""]

    def _match_plain(df, others):
        """Restrict an external 'plain' baseline (13_qwen_divergence.sh output,
        all groups and seeds) to the groups/seeds the key-fix cells ran on."""
        for col in ("group", "seed"):
            vals = {v for o in others if col in o.columns for v in o[col].unique()}
            if vals and col in df.columns:
                df = df[df[col].isin(vals)]
        # same sentences too (MAXPG=50 cells vs a 100-sentence baseline)
        pairs = {(g, i) for o in others if {"group", "idx"} <= set(o.columns)
                 for g, i in zip(o["group"], o["idx"])}
        if pairs and {"group", "idx"} <= set(df.columns):
            df = df[[(g, i) in pairs for g, i in zip(df["group"], df["idx"])]]
        return df

    # --- objective scores -------------------------------------------------
    score_rows = []
    cell_scores = {}
    for cell in CELLS:
        p = os.path.join(args.results, f"qwen_keyfix_{cell}_scores.csv")
        if cell == "plain" and args.plain_scores:
            p = args.plain_scores
        if os.path.exists(p):
            cell_scores[cell] = pd.read_csv(p)
    for cell, s in cell_scores.items():
        if "config" not in s.columns or "cer" not in s.columns:
            continue
        if cell == "plain" and args.plain_scores:
            s = _match_plain(s, [v for k, v in cell_scores.items() if k != "plain"])
        s = s[s.config != "fp16"] if cell != "plain" else s
        for cfg, g in s.groupby("config"):
            score_rows.append({
                "cell": cell, "config": cfg, "n": len(g),
                "cer": g.cer.mean(), "wer": g.wer.mean() if "wer" in g else float("nan"),
                "collapse_%": 100 * (g.cer > COLLAPSE_CER).mean(),
                "spk_sim": g.spk_sim.mean() if "spk_sim" in g else float("nan"),
            })
    if score_rows:
        t = pd.DataFrame(score_rows).sort_values(["config", "cell"])
        lines += ["## Objective metrics (Whisper large-v3; collapse = CER > 0.5)", "",
                  _md(t, ".3f"), ""]
    else:
        lines += ["_no score CSVs found (run with SCORE=1 or 20_score.sh)_", ""]

    # --- divergence -------------------------------------------------------
    div_frames = []
    for cell in CELLS:
        paths = sorted(glob.glob(os.path.join(args.results, f"qwen_keyfix_{cell}_seed*.csv")))
        if cell == "plain" and args.plain_div:
            paths = [args.plain_div]
        for p in paths:
            d = pd.read_csv(p)
            for c in ("key_group", "pre_rope"):
                if c not in d.columns:
                    d[c] = 0
            d["cell"] = cell
            div_frames.append(d)
    if args.plain_div and div_frames:
        others = [f for f in div_frames if f.cell.iloc[0] != "plain"]
        div_frames = [_match_plain(f, others) if f.cell.iloc[0] == "plain" else f for f in div_frames]
    if div_frames:
        d = pd.concat(div_frames, ignore_index=True)
        # The recorder hooks eager_attention_forward, which also fires inside
        # the code predictor (Qwen3TTSAttention, layer_idx 0-4, ~15 sub-steps
        # per talker step, its cache is never TurboQuant-compressed). Those
        # rows outnumber the talker's layers 0-4 rows 16:1 and are measuring
        # nothing we deploy. The code predictor's sequence never exceeds
        # num_code_groups (16) positions; the talker's prompt alone is longer.
        if args.talker_min_pos > 0:
            before = d.groupby("layer").size()
            d = d[d.pos >= args.talker_min_pos]
            after = d.groupby("layer").size()
            lines += [f"Divergence rows restricted to the talker (pos >= {args.talker_min_pos}); "
                      f"rows per layer before/after the filter, layers 0-4 vs 5: "
                      + ", ".join(f"L{l}: {int(before.get(l, 0))}->{int(after.get(l, 0))}" for l in (0, 4, 5)),
                      ""]
        d["config"] = [_label(*r) for r in zip(d.key_bits, d.value_bits, d.rw, d.key_group, d.pre_rope)]
        m = (d.groupby(["cell", "config"])[["attn_js", "attn_top1", "out_cos", "cos_k", "cos_v", "relmse_k"]]
             .mean().reset_index().sort_values(["config", "cell"]))
        lines += ["## Attention / KV divergence (talker layers; mean over layers, positions, sentences)", "",
                  _md(m, ".5f"), ""]
        k2 = d[(d.key_bits == 2) & (d.value_bits == 2) & (d.rw == 0)]
        if len(k2):
            prof = k2.groupby(["layer", "cell"]).attn_js.mean().unstack("cell")
            lines += ["## Per-layer attn_js for K2V2@0 (does the hotspot survive the fix?)", "",
                      _md(prof, ".5f", index=True), ""]
            top = prof.idxmax()
            lines += ["Peak layer per cell: " + ", ".join(f"{c}: {int(l)}" for c, l in top.items()), ""]
    else:
        lines += ["_no divergence CSVs found_", ""]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
