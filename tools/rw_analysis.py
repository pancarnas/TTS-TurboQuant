"""Residual-window (rw) significance analysis for VALL-E.

Answers: does the residual window actually contribute, or is its effect within
noise? Uses PER-SENTENCE PAIRED comparison — the same sentence rendered at
different rw is matched on (group, idx, arm, seed) — so the test isolates the
rw effect from sentence/seed variance.

Reads a wav-score CSV (from tools/score_wav_dir.py; columns include group, idx,
arm, seed, config, key_bits, value_bits, rw, cer, wer). AR configs only
(drops fp16 and any -nar/-both), long group excluded by default.

For each (key_bits, value_bits) and each rw pair (e.g. 0→64, 0→128, 64→128):
mean Δ, bootstrap CI, paired Wilcoxon (Holm-corrected across comparisons per
metric), and fraction-improved. Δ = metric(rw_hi) − metric(rw_lo); negative Δ
means the larger window helped. If the CI straddles 0 and the test is ns, rw
does not significantly contribute at that bit-width.

Plots (designed NOT to mislead): mean CER/WER vs rw as small multiples by key
bits (lines = value bits) with bootstrap CI bands, **y-axis anchored at 0**, and
the fp16 baseline as a labelled reference line; plus a paired-Δ box of the
widest rw jump centred on 0.

  python tools/rw_analysis.py --scores results/vallex_grid_scores.csv \
      --out results/rw_significance.md --img-dir results/figures
"""

from __future__ import annotations

import argparse
import itertools
import os
import re
import sys

_TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _TOOLS)
sys.path.insert(0, os.path.dirname(_TOOLS))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analyze_kv_benchmark import bootstrap_ci, holm_correct, sig_marker  # noqa: E402

try:
    from scipy.stats import wilcoxon
except Exception:  # noqa: BLE001 - scipy optional; degrade to no p-values
    wilcoxon = None

_AR_CFG = re.compile(r"^K(\d+)V(\d+)@(\d+)$")  # AR only: no -nar/-both suffix
PAIR_KEYS = ["group", "idx", "arm", "seed"]
METRICS = [("cer", "CER"), ("wer", "WER")]


def _load(scores: str, exclude_groups: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(scores)
    df = df[~df["group"].isin(exclude_groups)]
    fp16 = df[df["config"].astype(str).str.lower() == "fp16"].copy()
    ar = df[df["config"].astype(str).str.match(_AR_CFG)].copy()
    for c in ("key_bits", "value_bits", "rw"):
        ar[c] = ar[c].astype(int)
    return ar, fp16


def _paired_stats(ar: pd.DataFrame, metric: str) -> list[dict]:
    """Paired rw-vs-rw deltas per (key_bits, value_bits). Holm across all."""
    rows: list[dict] = []
    pvals: list[float] = []
    for (kb, vb), sub in ar.groupby(["key_bits", "value_bits"]):
        wide = sub.pivot_table(index=PAIR_KEYS, columns="rw", values=metric,
                               aggfunc="mean")
        rws = sorted(wide.columns)
        for r_lo, r_hi in itertools.combinations(rws, 2):
            paired = wide[[r_lo, r_hi]].dropna()
            n = len(paired)
            if n == 0:
                continue
            delta = (paired[r_hi] - paired[r_lo]).to_numpy()
            mean, lo, hi = bootstrap_ci(delta)
            if wilcoxon is None or np.allclose(delta, 0.0):
                p = float("nan") if wilcoxon is None else 1.0
            else:
                try:
                    p = float(wilcoxon(paired[r_hi], paired[r_lo]).pvalue)
                except ValueError:
                    p = 1.0
            rows.append({
                "key_bits": kb, "value_bits": vb,
                "cmp": f"rw{r_lo}->rw{r_hi}", "n": n,
                "mean_delta": mean, "ci_lo": lo, "ci_hi": hi,
                "frac_improved": float((delta < 0).mean()), "p": p,
            })
            pvals.append(p if not np.isnan(p) else 1.0)
    adj = holm_correct(pvals) if pvals else []
    for r, pa in zip(rows, adj):
        r["p_holm"] = pa
    return rows


def _fmt(x, nd=4):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def _md_table(rows: list[dict], metric_label: str) -> list[str]:
    head = ("| K | V | comparison | n | mean Δ | 95% CI | frac improved | "
            "p (Holm) | sig |")
    out = [f"### {metric_label}", "", head, "|" + "---|" * 9]
    for r in sorted(rows, key=lambda d: (-d["key_bits"], -d["value_bits"], d["cmp"])):
        ci = f"[{_fmt(r['ci_lo'])}, {_fmt(r['ci_hi'])}]"
        pa = r.get("p_holm", float("nan"))
        out.append(
            f"| K{r['key_bits']} | V{r['value_bits']} | {r['cmp']} | {r['n']} | "
            f"{r['mean_delta']:+.4f} | {ci} | {r['frac_improved']:.2f} | "
            f"{_fmt(pa)} | {sig_marker(pa) if not np.isnan(pa) else 'n/a'} |"
        )
    out.append("")
    return out


def _rw_effect_fig(ar: pd.DataFrame, fp16: pd.DataFrame, metric: str,
                   label: str, out: str) -> None:
    """Small multiples by key_bits; lines = value_bits; x = rw; CI bands;
    y anchored at 0; fp16 reference line. Anti-misleading by construction."""
    kbs = sorted(ar["key_bits"].unique(), reverse=True)
    fig, axes = plt.subplots(1, len(kbs), figsize=(4 * len(kbs) + 1, 4),
                             squeeze=False, sharey=True)
    fp16_mean = float(fp16[metric].mean()) if not fp16.empty else None
    ymax = ar[metric].max()
    for ax, kb in zip(axes[0], kbs):
        sub = ar[ar["key_bits"] == kb]
        for vb in sorted(sub["value_bits"].unique(), reverse=True):
            g = sub[sub["value_bits"] == vb]
            rws = sorted(g["rw"].unique())
            means, los, his = [], [], []
            for rw in rws:
                m, lo, hi = bootstrap_ci(g[g["rw"] == rw][metric].to_numpy())
                means.append(m); los.append(lo); his.append(hi)
            ax.plot(rws, means, marker="o", label=f"V{vb}")
            ax.fill_between(rws, los, his, alpha=0.15)
        if fp16_mean is not None:
            ax.axhline(fp16_mean, ls="--", color="#888",
                       label=f"fp16 ({fp16_mean:.3f})")
        ax.set_title(f"K{kb}", fontweight="bold")
        ax.set_xlabel("residual window (tokens)")
        ax.set_xticks(sorted(ar["rw"].unique()))
        ax.set_ylim(0, ymax * 1.15)  # anchored at 0 — no truncated axis
        ax.grid(True, alpha=0.25)
    axes[0][0].set_ylabel(label)
    axes[0][-1].legend(fontsize=8, loc="best")
    fig.suptitle(f"{label} vs residual window (VALL-E, mean ± 95% CI; y from 0)",
                 fontsize=12, fontweight="bold")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def _paired_delta_fig(ar: pd.DataFrame, metric: str, label: str, out: str) -> None:
    """Box of per-sentence Δ for the widest rw jump, per (K,V), centred on 0."""
    rws = sorted(ar["rw"].unique())
    if len(rws) < 2:
        return
    r_lo, r_hi = rws[0], rws[-1]
    data, labels = [], []
    for (kb, vb), sub in ar.groupby(["key_bits", "value_bits"]):
        wide = sub.pivot_table(index=PAIR_KEYS, columns="rw", values=metric,
                               aggfunc="mean")
        if r_lo not in wide.columns or r_hi not in wide.columns:
            continue
        paired = wide[[r_lo, r_hi]].dropna()
        if paired.empty:
            continue
        data.append((paired[r_hi] - paired[r_lo]).to_numpy())
        labels.append(f"K{kb}V{vb}")
    if not data:
        return
    fig, ax = plt.subplots(figsize=(1.1 * len(data) + 2, 4.5))
    ax.axhline(0, color="#d00", lw=1.5, zorder=1)
    ax.boxplot(data, showfliers=False, zorder=2)
    ax.set_xticks(range(1, len(labels) + 1), labels, rotation=45, ha="right")
    ax.set_ylabel(f"Δ{label}  (rw{r_hi} − rw{r_lo}, per sentence)")
    ax.set_title(
        f"Per-sentence {label} change from rw{r_lo}→rw{r_hi} (VALL-E)\n"
        "straddling 0 ⇒ residual window does not contribute",
        fontsize=11, fontweight="bold",
    )
    ax.grid(True, axis="y", alpha=0.25)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", default="results/vallex_wav_scores.csv")
    ap.add_argument("--exclude-groups", default="libritts_long")
    ap.add_argument("--out", default="results/rw_significance.md")
    ap.add_argument("--img-dir", default="results/figures")
    args = ap.parse_args()

    excl = [g.strip() for g in args.exclude_groups.split(",") if g.strip()]
    ar, fp16 = _load(args.scores, excl)
    if ar.empty:
        raise SystemExit("no AR-config rows after filtering")

    lines = ["# Residual-window significance (VALL-E, per-sentence paired)", ""]
    if wilcoxon is None:
        lines.append("_scipy unavailable — p-values omitted (Δ/CI still valid)._\n")
    lines.append(
        "Δ = metric(rw_hi) − metric(rw_lo); negative ⇒ larger window helped. "
        "A CI straddling 0 with a non-significant test ⇒ rw does not contribute.\n"
    )
    for metric, label in METRICS:
        rows = _paired_stats(ar, metric)
        lines += _md_table(rows, label)
        _rw_effect_fig(ar, fp16, metric, label,
                       os.path.join(args.img_dir, f"rw_effect_{metric}.png"))
    _paired_delta_fig(ar, "wer", "WER",
                      os.path.join(args.img_dir, "rw_paired_delta_wer.png"))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
