"""Key-bits × value-bits grids from the combined summary.

Reads the tidy per-(model, group, config) table written by
tools/combined_summary.py and pivots each metric into a key_bits × value_bits
grid, per dataset and averaged across datasets — the deeper cut that shows the
bit-allocation dependence with the noisy axes (residual window, stage)
marginalized out.

Defaults: AR-stage configs only (the ``-nar`` / ``-both`` arms are a separate
axis), fp16 excluded (no bits), and each cell is the mean over the residual
windows present for that (key, value) pair. The across-dataset row is a macro
mean (simple mean of the per-dataset cell means, so a small group like
ellav_hard is not down-weighted).

  python tools/kv_grid.py --summary results/combined_summary.csv \
      --out results/kv_grids.md

Metrics: CER, WER, cos_k, cos_v. Cells are '—' where no config supplies them
(e.g. Qwen WER before the re-score; VALL-E cos_* outside librispeech_pc).
"""

from __future__ import annotations

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

_CFG_RE = re.compile(r"^K(\d+)V(\d+)@(\d+)$")  # AR-only (no -nar/-both suffix)
METRICS = [("cer", "CER"), ("wer", "WER"), ("cos_k", "cos_k"), ("cos_v", "cos_v")]


def _parse_bits(df: pd.DataFrame) -> pd.DataFrame:
    kb, vb = [], []
    for c in df["config"]:
        m = _CFG_RE.match(str(c))
        kb.append(int(m.group(1)) if m else None)
        vb.append(int(m.group(2)) if m else None)
    out = df.copy()
    out["key_bits"] = kb
    out["value_bits"] = vb
    return out[out["key_bits"].notna()].astype({"key_bits": int, "value_bits": int})


def _grid_md(pivot: pd.DataFrame, metric: str) -> list[str]:
    """Render a key×value pivot as a markdown table (rows=key bits desc)."""
    vcols = sorted(pivot.columns, reverse=True)
    krows = sorted(pivot.index, reverse=True)
    head = f"| {metric}  (K↓ / V→) | " + " | ".join(f"V{v}" for v in vcols) + " |"
    sep = "|---" * (len(vcols) + 1) + "|"
    lines = [head, sep]
    for k in krows:
        cells = []
        for v in vcols:
            x = pivot.loc[k, v] if v in pivot.columns else None
            cells.append("—" if x is None or pd.isna(x) else f"{x:.4f}")
        lines.append(f"| K{k} | " + " | ".join(cells) + " |")
    return lines


def build(summary_path: str) -> pd.DataFrame:
    df = pd.read_csv(summary_path)
    df = df[df["group"] != "smoke"]
    return _parse_bits(df)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", default="results/combined_summary.csv")
    ap.add_argument("--out", default="results/kv_grids.md")
    args = ap.parse_args()

    df = build(args.summary)
    lines = ["# Key-bits × value-bits grids (AR-only, averaged over residual window)", ""]

    for metric, label in METRICS:
        lines.append(f"## {label}")
        lines.append("")
        for model in sorted(df["model"].unique()):
            m = df[df["model"] == model]
            lines.append(f"### {model}")
            lines.append("")
            groups = sorted(m["group"].unique())
            per_group_pivots = []
            for grp in groups:
                sub = m[m["group"] == grp]
                piv = sub.pivot_table(
                    index="key_bits", columns="value_bits",
                    values=metric, aggfunc="mean",
                )
                per_group_pivots.append(piv)
                lines.append(f"**{grp}**")
                lines += _grid_md(piv, label)
                lines.append("")
            # Macro mean across datasets: average the per-group cell means.
            if per_group_pivots:
                mean_piv = pd.concat(per_group_pivots).groupby(level=0).mean()
                lines.append("**mean over datasets**")
                lines += _grid_md(mean_piv, label)
                lines.append("")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
