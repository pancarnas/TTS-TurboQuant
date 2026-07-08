"""Length-stratified metrics from combined_summary.csv.

Splits the eval datasets by sequence length and reports:

  * LONG  — libritts_long (VALL-E only; Qwen never ran long sequences),
            per config.
  * SHORT/MID — seedtts_en + librispeech_pc + ellav_hard, averaged over those
            three datasets, per model per config (Qwen and VALL-E side by side).

Metrics: CER, WER, collapse%, spkSim, cos_k, cos_v. Blank ('—') where a model
lacks the metric (Qwen WER, cos_* outside covered datasets). Writes a markdown
file and prints the same.

  python tools/length_summary.py --summary results/combined_summary.csv \
      --out results/length_summary.md
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

SHORT_MID = ["seedtts_en", "librispeech_pc", "ellav_hard"]
LONG = ["libritts_long"]
METRIC_COLS = [
    ("cer", "CER"), ("wer", "WER"), ("collapse_pct", "collapse%"),
    ("spk_sim", "spkSim"), ("cos_k", "cos_k"), ("cos_v", "cos_v"),
]


def _fmt(col: str, v) -> str:
    if pd.isna(v):
        return "—"
    return f"{v:.1f}" if col == "collapse_pct" else f"{v:.4f}"


def _table(rows: pd.DataFrame, id_cols: list[str]) -> list[str]:
    labels = id_cols + [lab for _, lab in METRIC_COLS]
    out = ["| " + " | ".join(labels) + " |", "|" + "---|" * len(labels)]
    for _, r in rows.iterrows():
        cells = [str(r[c]) for c in id_cols]
        cells += [_fmt(c, r[c]) for c, _ in METRIC_COLS]
        out.append("| " + " | ".join(cells) + " |")
    return out


def _render_png(rows: pd.DataFrame, id_cols: list[str], out: str, title: str) -> None:
    """Styled table image of `rows` (id_cols + the metric columns)."""
    if rows.empty:
        print(f"  skip {out}: no rows")
        return
    header = id_cols + [lab for _, lab in METRIC_COLS]
    text = [
        [str(r[c]) for c in id_cols] + [_fmt(c, r[c]) for c, _ in METRIC_COLS]
        for _, r in rows.iterrows()
    ]
    n = len(text)
    fig, ax = plt.subplots(figsize=(11, 0.34 * (n + 1) + 0.8))
    ax.axis("off")
    tbl = ax.table(cellText=text, colLabels=header, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.3)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if row == 0:
            cell.set_facecolor("#40466e")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#f4f5fa" if row % 2 else "#ffffff")
        if col < len(id_cols):
            cell.set_text_props(fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  ({n} rows)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", default="results/combined_summary.csv")
    ap.add_argument("--out", default="results/length_summary.md")
    ap.add_argument(
        "--img-dir", default="results/figures/tables",
        help="Also write long.png and short_mid.png table images here.",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.summary)
    df = df[df["group"] != "smoke"]
    metric_names = [c for c, _ in METRIC_COLS]

    lines = ["# Length-stratified metrics", ""]

    # LONG — per (model, config); in practice VALL-E only.
    lines += ["## Long sequences — libritts_long", ""]
    lg = df[df["group"].isin(LONG)].copy()
    models_long = sorted(lg["model"].unique())
    lines.append(f"_models with long-sequence data: {', '.join(models_long) or 'none'}_")
    lines.append("")
    lg = lg.sort_values(["model", "config"])
    lines += _table(lg, ["model", "config"])
    lines.append("")

    # SHORT/MID — averaged over the three standard datasets, per model/config.
    lines += ["## Short / mid sequences — mean over seedtts_en, librispeech_pc, ellav_hard", ""]
    sm = df[df["group"].isin(SHORT_MID)]
    agg = (
        sm.groupby(["model", "config"])[metric_names]
        .mean()
        .reset_index()
        .sort_values(["model", "config"])
    )
    lines += _table(agg, ["model", "config"])
    lines.append("")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote {args.out}")

    if args.img_dir:
        _render_png(
            lg, ["model", "config"],
            os.path.join(args.img_dir, "length_long.png"),
            "Long sequences — libritts_long (VALL-E only)",
        )
        _render_png(
            agg, ["model", "config"],
            os.path.join(args.img_dir, "length_short_mid.png"),
            "Short / mid — mean over seedtts_en, librispeech_pc, ellav_hard",
        )


if __name__ == "__main__":
    main()
