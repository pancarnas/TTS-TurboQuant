"""Multi-seed error bars for the headline configs (CER / WER / spk_sim).

Reads one or more labelled score CSVs (from tools/score_wav_dir.py, each with a
`seed` column) and reports, per config, the mean and between-seed spread — the
error bars that tell whether a config-to-config difference (K4V2 vs K3V3, or
pl2 vs pl0) is real or within sampling noise.

Unit of variability = the SEED: for each config we take the per-seed mean over
sentences (one number per seed), then report mean +/- std across seeds and a
bootstrap CI over sentence x seed. A difference smaller than the error bars is
not claimable.

  python tools/aggregate_seeds.py \
      --scores pl0=results/seed_pl0_scores.csv pl2=results/seed_pl2_scores.csv \
      --exclude-groups ellav_hard --out results/seed_errorbars.md \
      --img results/figures/seed_errorbars.png
"""

from __future__ import annotations

import argparse
import os
import sys

_TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _TOOLS)
sys.path.insert(0, os.path.dirname(_TOOLS))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analyze_kv_benchmark import bootstrap_ci  # noqa: E402

METRICS = [("cer", "CER"), ("wer", "WER"), ("spk_sim", "spkSim")]


def _config_order(configs):
    """fp16 first, then by (key desc, value desc, rw asc)."""
    import re
    def key(c):
        if str(c).lower() == "fp16":
            return (-999, 0, 0)
        m = re.match(r"K(\d+)V(\d+)@(\d+)", str(c))
        if not m:
            return (0, 0, 0)
        return (-int(m.group(1)), -int(m.group(2)), int(m.group(3)))
    return sorted(configs, key=key)


def _stats(path: str, exclude) -> pd.DataFrame:
    d = pd.read_csv(path)
    d = d[d["group"] != "smoke"]
    if exclude:
        d = d[~d["group"].isin(exclude)]
    rows = []
    for cfg, g in d.groupby("config"):
        row = {"config": cfg, "n_seeds": int(g["seed"].nunique())}
        for m, _ in METRICS:
            if m not in g or g[m].isna().all():
                row[f"{m}_mean"] = np.nan
                row[f"{m}_std"] = np.nan
                continue
            # per-seed mean over sentences, then spread across seeds
            seed_means = g.groupby("seed")[m].mean()
            row[f"{m}_mean"] = float(seed_means.mean())
            row[f"{m}_std"] = float(seed_means.std(ddof=1)) if len(seed_means) > 1 else 0.0
            _, lo, hi = bootstrap_ci(g[m].to_numpy())
            row[f"{m}_lo"], row[f"{m}_hi"] = lo, hi
        rows.append(row)
    out = pd.DataFrame(rows)
    out["__ord"] = out["config"].map({c: i for i, c in enumerate(_config_order(out["config"]))})
    return out.sort_values("__ord").drop(columns="__ord").reset_index(drop=True)


def _md(series: list[tuple[str, pd.DataFrame]]) -> list[str]:
    lines = ["# Multi-seed error bars (mean +/- std across seeds)", ""]
    for label, df in series:
        lines.append(f"## {label}")
        lines.append("")
        lines.append("| config | seeds | CER | WER | spkSim |")
        lines.append("|---|---|---|---|---|")
        for _, r in df.iterrows():
            def cell(m):
                mu, sd = r.get(f"{m}_mean"), r.get(f"{m}_std")
                if pd.isna(mu):
                    return "—"
                return f"{mu:.3f} ± {sd:.3f}"
            lines.append(
                f"| {r['config']} | {int(r['n_seeds'])} | "
                f"{cell('cer')} | {cell('wer')} | {cell('spk_sim')} |"
            )
        lines.append("")
    return lines


def _plot(series, metric, label, out):
    configs = _config_order(
        set().union(*[set(df["config"]) for _, df in series])
    )
    x = np.arange(len(configs))
    w = 0.8 / max(len(series), 1)
    fig, ax = plt.subplots(figsize=(1.1 * len(configs) + 2, 4.5))
    for si, (name, df) in enumerate(series):
        d = df.set_index("config")
        means = [d[f"{metric}_mean"].get(c, np.nan) for c in configs]
        stds = [d[f"{metric}_std"].get(c, np.nan) for c in configs]
        ax.bar(x + si * w - 0.4 + w / 2, means, w, yerr=stds, capsize=3,
               label=name)
    ax.set_xticks(x, configs, rotation=45, ha="right")
    ax.set_ylabel(label)
    ax.set_ylim(bottom=0)  # anchored at 0 — honest error-bar scale
    ax.set_title(f"{label} by config (mean +/- std across seeds)",
                 fontweight="bold")
    ax.legend(title="protection" if len(series) > 1 else None)
    ax.grid(True, axis="y", alpha=0.25)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", nargs="+", required=True, metavar="label=path")
    ap.add_argument("--exclude-groups", default="ellav_hard")
    ap.add_argument("--out", default="results/seed_errorbars.md")
    ap.add_argument("--img", default="results/figures/seed_errorbars.png")
    args = ap.parse_args()

    excl = [g.strip() for g in args.exclude_groups.split(",") if g.strip()]
    series = []
    for spec in args.scores:
        label, path = spec.split("=", 1)
        series.append((label, _stats(path, excl)))

    lines = _md(series)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote {args.out}")

    base, ext = os.path.splitext(args.img)
    for metric, label in METRICS[:2]:  # CER + WER figures
        _plot(series, metric, label, f"{base}_{metric}{ext}")


if __name__ == "__main__":
    main()
