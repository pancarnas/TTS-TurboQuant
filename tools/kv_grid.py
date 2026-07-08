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
    kb, vb, rw = [], [], []
    for c in df["config"]:
        m = _CFG_RE.match(str(c))
        kb.append(int(m.group(1)) if m else None)
        vb.append(int(m.group(2)) if m else None)
        rw.append(int(m.group(3)) if m else None)
    out = df.copy()
    out["key_bits"] = kb
    out["value_bits"] = vb
    out["rw"] = rw
    return out[out["key_bits"].notna()].astype(
        {"key_bits": int, "value_bits": int, "rw": int}
    )


def _numkey(label: str) -> tuple:
    """Sort key from a label's numbers: bits desc, then rw asc. 'K4rw64'->(-4,64)."""
    nums = [int(x) for x in re.findall(r"\d+", str(label))]
    bits = nums[0] if nums else 0
    rw = nums[1] if len(nums) > 1 else 0
    return (-bits, rw)


def _grid_md(pivot: pd.DataFrame, metric: str) -> list[str]:
    """Render a pivot as a markdown table; row/col labels are strings."""
    rows = sorted(pivot.index, key=_numkey)
    cols = sorted(pivot.columns, key=_numkey)
    head = f"| {metric}  (K↓ / V→) | " + " | ".join(str(c) for c in cols) + " |"
    sep = "|---" * (len(cols) + 1) + "|"
    lines = [head, sep]
    for r in rows:
        cells = []
        for c in cols:
            x = pivot.loc[r, c] if c in pivot.columns else None
            cells.append("—" if x is None or pd.isna(x) else f"{x:.4f}")
        lines.append(f"| {r} | " + " | ".join(cells) + " |")
    return lines


def build(summary_path: str) -> pd.DataFrame:
    df = pd.read_csv(summary_path)
    df = df[df["group"] != "smoke"]
    return _parse_bits(df)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", default="results/combined_summary.csv")
    ap.add_argument("--out", default="results/kv_grids.md")
    ap.add_argument(
        "--overall",
        action="store_true",
        help="One grid per model using the mean over all datasets only "
        "(skip the per-dataset grids).",
    )
    ap.add_argument(
        "--with-rw",
        action="store_true",
        help="Fold the residual window into the axes: rows K{bits}rw{rw}, "
        "cols V{bits}rw{rw}. NOTE rw is shared by K and V in a config, so only "
        "rw-matched cells fill (block-diagonal); off-diagonal is '—'.",
    )
    args = ap.parse_args()

    df = build(args.summary)
    if args.with_rw:
        df["rowkey"] = "K" + df["key_bits"].astype(str) + "rw" + df["rw"].astype(str)
        df["colkey"] = "V" + df["value_bits"].astype(str) + "rw" + df["rw"].astype(str)
    else:
        df["rowkey"] = "K" + df["key_bits"].astype(str)
        df["colkey"] = "V" + df["value_bits"].astype(str)
    title = "# Key-bits × value-bits grids (AR-only, averaged over residual window"
    title += ", mean over datasets)" if args.overall else ")"
    lines = [title, ""]

    for metric, label in METRICS:
        lines.append(f"## {label}")
        lines.append("")
        for model in sorted(df["model"].unique()):
            m = df[df["model"] == model]
            lines.append(f"### {model}")
            lines.append("")
            per_group_pivots = []
            for grp in sorted(m["group"].unique()):
                sub = m[m["group"] == grp]
                piv = sub.pivot_table(
                    index="rowkey", columns="colkey",
                    values=metric, aggfunc="mean",
                )
                per_group_pivots.append(piv)
                if not args.overall:
                    lines.append(f"**{grp}**")
                    lines += _grid_md(piv, label)
                    lines.append("")
            # Macro mean across datasets: average the per-group cell means.
            if per_group_pivots:
                mean_piv = pd.concat(per_group_pivots).groupby(level=0).mean()
                if not args.overall:
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
