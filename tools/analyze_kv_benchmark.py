"""Statistical analysis of the rigorous KV-quantization benchmark.

Reads the decoupled CSV artifacts produced by the benchmark harness and answers
the central question — *what is caused by compression vs by sampling randomness?*
— with confidence intervals, paired significance tests, and a variance
decomposition. It performs NO generation; it is pure pandas/scipy so it is fully
reproducible from the CSVs alone.

Inputs (auto-discovered as the most recent match under ``results/``, or passed
explicitly):
  - per-trial CSV: ``*_trials_*.csv`` (one row per arm × sentence × seed × config)
  - KV-reconstruction CSV: ``*_kv_recon_*.csv`` (per layer; deterministic)

Outputs (printed; optionally written to markdown with ``--out-md``):
  1. Mean ± 95% bootstrap CI of CER / speaker-sim per (arm, group, config).
  2. Paired Wilcoxon (Holm-corrected) of each config vs baseline on matched seeds.
  3. Variance decomposition of CER into sentence / seed / config (one-way η²).
  4. Intrinsic compression ranking from KV reconstruction (no sampling noise).
  5. Greedy-vs-sampling contrast — how much of the CER bouncing was noise.

Usage:
    python tools/analyze_kv_benchmark.py [--results-dir results]
        [--trials PATH] [--kv-recon PATH] [--out-md report.md] [--n-boot 10000]
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
except ImportError:  # scipy is a hard dep of turboquant, but degrade gracefully
    wilcoxon = None


BASELINE_HINT = "baseline"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _latest(results_dir: str, suffix: str) -> Optional[str]:
    """Most recently modified ``results/*<suffix>`` file, or None."""
    matches = glob.glob(os.path.join(results_dir, f"*{suffix}"))
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def load_trials(
    trials: Optional[str], trials_glob: Optional[str], results_dir: str
) -> tuple[Optional[pd.DataFrame], list[str]]:
    """Load the per-trial frame, concatenating shard CSVs when a glob is given.

    Precedence: explicit ``--trials-glob`` (concat every match — this is how a
    parallel sharded run is reassembled) > explicit ``--trials`` (one file) >
    auto-discover the latest ``*_trials_*.csv``. Returns (df, paths_used) so the
    set of concatenated files is reported, never silently truncated.
    """
    if trials_glob:
        paths = sorted(glob.glob(trials_glob))
        if not paths:
            return None, []
        frames = [pd.read_csv(p) for p in paths]
        return pd.concat(frames, ignore_index=True), paths
    path = trials or _latest(results_dir, "_trials_*.csv")
    if path and os.path.exists(path):
        return pd.read_csv(path), [path]
    return None, []


def _is_baseline(df: pd.DataFrame) -> pd.Series:
    """A row is baseline when it carries no key_bits (the uncompressed config)."""
    return df["key_bits"].isna()


def _baseline_name(df: pd.DataFrame) -> Optional[str]:
    base = df.loc[_is_baseline(df), "config"].unique()
    if len(base):
        return base[0]
    # fall back to a name containing "baseline"
    for c in df["config"].unique():
        if BASELINE_HINT in str(c).lower():
            return c
    return None


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def bootstrap_ci(
    values, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Return (mean, lo, hi) with a percentile bootstrap CI. NaNs dropped.

    Bootstrap rather than a t-interval because CER is bounded and floor-skewed,
    so the sampling distribution of the mean is not symmetric near zero.
    """
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return (np.nan, np.nan, np.nan)
    if x.size == 1:
        return (float(x[0]), float(x[0]), float(x[0]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (float(x.mean()), lo, hi)


def holm_correct(pvals: list[float]) -> list[float]:
    """Holm–Bonferroni step-down adjusted p-values (same order as input)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(running, 1.0)
    return adj


def sig_marker(p: float) -> str:
    if np.isnan(p):
        return "n/a"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def eta_squared(df: pd.DataFrame, value: str, factors: list[str]) -> dict[str, float]:
    """One-way η² (marginal share of variance) per factor for ``value``.

    NOTE: marginal and non-orthogonal — factors can overlap, so shares need not
    sum to 1. The point is the comparison: if config's share is tiny next to
    seed + sentence, the headline CER trend is dominated by noise, not compression.
    """
    x = df[value].dropna()
    if x.empty:
        return {f: float("nan") for f in factors}
    grand = x.mean()
    ss_total = float(((x - grand) ** 2).sum())
    out = {}
    for f in factors:
        ss = 0.0
        for _, grp in df.dropna(subset=[value]).groupby(f):
            ss += len(grp) * (grp[value].mean() - grand) ** 2
        out[f] = (ss / ss_total) if ss_total > 0 else float("nan")
    return out


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------


def _emit(lines: list[str], text: str = "") -> None:
    print(text)
    lines.append(text)


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Column ``name`` if present, else an empty float series (CSV back-compat)."""
    if name in df.columns:
        return df[name]
    return pd.Series(dtype=float)


def _normalize_temperature(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a string ``temperature`` column exists. Blank/NaN -> 'default'.

    Keeps the analysis backward-compatible with pre-sweep CSVs (no column) and
    with greedy/default rows (blank temperature), so grouping never drops rows.
    """
    if "temperature" not in df.columns:
        df["temperature"] = "default"
        return df

    def _bucket(v: Any) -> str:
        if pd.isna(v) or str(v).strip() == "":
            return "default"
        try:
            return f"{float(v):g}"
        except (TypeError, ValueError):
            return str(v)

    df["temperature"] = df["temperature"].apply(_bucket)
    return df


def section_means(df: pd.DataFrame, n_boot: int, lines: list[str]) -> None:
    _emit(lines, "\n## 1. Mean ± 95% bootstrap CI (CER, SpkSim)\n")
    for arm in sorted(df["arm"].dropna().unique()):
        _emit(lines, f"### arm: {arm}")
        sub = df[df["arm"] == arm]
        _emit(
            lines,
            f"{'group':<10} {'temp':>7} {'config':<22} {'n':>4}  "
            f"{'CER mean':>9} {'[95% CI]':>20}   {'SpkSim':>8} {'SpkSimRef':>10}",
        )
        for group in sorted(sub["group"].dropna().unique()):
            for temp in sorted(sub[sub["group"] == group]["temperature"].unique()):
                cells = sub[(sub["group"] == group) & (sub["temperature"] == temp)]
                for config in cells["config"].unique():
                    cell = cells[cells["config"] == config]
                    cm, clo, chi = bootstrap_ci(cell["cer"], n_boot)
                    sm, _, _ = bootstrap_ci(cell["spk_sim"], n_boot)
                    smr, _, _ = bootstrap_ci(_col(cell, "spk_sim_ref"), n_boot)
                    ci = f"[{clo:.4f}, {chi:.4f}]"
                    sm_str = "---" if np.isnan(sm) else f"{sm:.4f}"
                    smr_str = "---" if np.isnan(smr) else f"{smr:.4f}"
                    _emit(
                        lines,
                        f"{group:<10} {str(temp):>7} {config:<22} {len(cell):>4}  "
                        f"{cm:>9.4f} {ci:>20}   {sm_str:>8} {smr_str:>10}",
                    )
        _emit(lines)


def section_paired_tests(df: pd.DataFrame, lines: list[str]) -> None:
    _emit(
        lines, "\n## 2. Paired Wilcoxon vs baseline (matched seeds, Holm-corrected)\n"
    )
    base = _baseline_name(df)
    if base is None:
        _emit(lines, "  (no baseline config found — skipping)")
        return
    if wilcoxon is None:
        _emit(lines, "  (scipy unavailable — skipping)")
        return

    for arm in sorted(df["arm"].dropna().unique()):
        sub = df[df["arm"] == arm]
        # paired on (group, idx, seed); columns = config, value = cer
        wide = sub.pivot_table(
            index=["group", "idx", "seed"],
            columns="config",
            values="cer",
            aggfunc="mean",
        )
        if base not in wide.columns:
            continue
        configs = [c for c in wide.columns if c != base]
        rows, pvals = [], []
        for c in configs:
            paired = wide[[base, c]].dropna()
            if len(paired) < 1:
                rows.append((c, len(paired), np.nan, np.nan))
                pvals.append(1.0)
                continue
            delta = (paired[c] - paired[base]).to_numpy()
            median_delta = float(np.median(delta))
            if np.allclose(delta, 0.0):
                p = 1.0  # wilcoxon undefined when every pair is identical
            else:
                try:
                    p = float(wilcoxon(paired[c], paired[base]).pvalue)
                except ValueError:
                    p = 1.0
            rows.append((c, len(paired), median_delta, p))
            pvals.append(p)
        adj = holm_correct(pvals)

        _emit(lines, f"### arm: {arm}  (baseline = {base})")
        _emit(
            lines,
            f"{'config':<22} {'n_pairs':>7} {'median ΔCER':>12} "
            f"{'p_raw':>9} {'p_holm':>9}  sig",
        )
        for (c, n, md, p), pa in zip(rows, adj):
            md_str = "n/a" if np.isnan(md) else f"{md:+.4f}"
            p_str = "n/a" if np.isnan(p) else f"{p:.4f}"
            _emit(
                lines,
                f"{c:<22} {n:>7} {md_str:>12} {p_str:>9} {pa:>9.4f}  {sig_marker(pa)}",
            )
        _emit(lines)


def section_variance(df: pd.DataFrame, lines: list[str]) -> None:
    _emit(lines, "\n## 3. CER variance decomposition (one-way η², marginal)\n")
    _emit(lines, "  Headline: if config's share is small next to seed + sentence,")
    _emit(
        lines, "  the across-config CER trend is dominated by noise, not compression.\n"
    )
    for arm in sorted(df["arm"].dropna().unique()):
        sub = df[df["arm"] == arm].dropna(subset=["cer"])
        if sub.empty:
            continue
        factors = ["config", "sentence_hash", "seed", "temperature"]
        shares = eta_squared(sub, "cer", factors)
        _emit(lines, f"### arm: {arm}")
        for factor, label in [
            ("config", "config (compression)"),
            ("sentence_hash", "sentence"),
            ("seed", "seed (sampling noise)"),
            ("temperature", "temperature (sampling regime)"),
        ]:
            _emit(lines, f"  {label:<32} η² = {shares[factor]:.3f}")
        _emit(lines)


def section_intrinsic(kv: pd.DataFrame, lines: list[str]) -> None:
    _emit(
        lines,
        "\n## 4. Intrinsic compression ranking — KV reconstruction (no sampling noise)\n",
    )
    g = (
        kv.groupby(["config", "variant"])
        .agg(
            cos_k=("cos_k", "mean"),
            cos_v=("cos_v", "mean"),
            relmse_k=("relmse_k", "mean"),
            relmse_v=("relmse_v", "mean"),
        )
        .reset_index()
    )
    _emit(
        lines,
        f"{'config':<14} {'variant':<11} {'cos_k':>8} {'cos_v':>8} "
        f"{'relmse_k':>10} {'relmse_v':>10}",
    )
    for _, r in g.iterrows():
        _emit(
            lines,
            f"{r['config']:<14} {r['variant']:<11} {r['cos_k']:>8.4f} "
            f"{r['cos_v']:>8.4f} {r['relmse_k']:>10.4g} {r['relmse_v']:>10.4g}",
        )
    _emit(lines)


def section_arm_contrast(df: pd.DataFrame, lines: list[str]) -> None:
    arms = set(df["arm"].dropna().unique())
    if not {"greedy", "sampling"}.issubset(arms):
        return
    _emit(lines, "\n## 5. Greedy vs sampling contrast (mean CER by config)\n")
    pivot = df.pivot_table(index="config", columns="arm", values="cer", aggfunc="mean")
    _emit(lines, f"{'config':<22} {'greedy':>9} {'sampling':>9} {'Δ(samp-greedy)':>15}")
    for config, row in pivot.iterrows():
        gd = row.get("greedy", np.nan)
        sp = row.get("sampling", np.nan)
        d = sp - gd
        _emit(lines, f"{config:<22} {gd:>9.4f} {sp:>9.4f} {d:>15.4f}")
    _emit(lines, "\n  Greedy isolates compression (deterministic decode); a large")
    _emit(lines, "  sampling-minus-greedy gap is sampling noise, not compression.")


def section_temperature_trend(df: pd.DataFrame, n_boot: int, lines: list[str]) -> None:
    """CER (mean ± CI) per config across the swept temperatures (sampling arm).

    Answers the reviewer's question directly: at which temperature does each
    compressed config start to diverge from baseline? Emitted only when the
    sampling arm actually carries more than one temperature.
    """
    if "sampling" not in set(df["arm"].dropna().unique()):
        return
    samp = df[df["arm"] == "sampling"]
    temps = sorted(t for t in samp["temperature"].unique() if t != "default")
    if len(temps) < 2:
        return

    _emit(
        lines,
        "\n## 6. CER vs temperature (sampling arm, pooled over sentences × seeds)\n",
    )
    _emit(
        lines, "  Higher temperature flattens sampling; compression-induced divergence"
    )
    _emit(lines, "  should grow with temperature if compression is actually applied.\n")
    header = f"{'config':<22}" + "".join(f"{('T=' + t):>14}" for t in temps)
    _emit(lines, header)
    for config in samp["config"].unique():
        row = f"{config:<22}"
        for t in temps:
            cell = samp[(samp["config"] == config) & (samp["temperature"] == t)]
            cm, clo, chi = bootstrap_ci(cell["cer"], n_boot)
            row += (
                f"{f'{cm:.3f}[{clo:.3f},{chi:.3f}]':>14}"
                if not np.isnan(cm)
                else f"{'---':>14}"
            )
        _emit(lines, row)


def section_length_trend(df: pd.DataFrame, n_boot: int, lines: list[str]) -> None:
    """CER (mean ± CI) per config across decode-length buckets.

    The project's headline long-context result: KV-compression error grows with
    sequence length (more cache to quantize). Buckets ``n_ar_tokens`` into terciles
    so you can read off whether the gap to baseline widens for longer decodes.
    Emitted only when there is a real spread of decode lengths.
    """
    if "n_ar_tokens" not in df.columns:
        return
    lengths = pd.to_numeric(df["n_ar_tokens"], errors="coerce")
    if lengths.dropna().nunique() < 3:
        return
    # Bucket by RANK so ties (e.g. many identical short decodes) never collapse the
    # quantile edges; this yields up to 3 equal-count groups ordered by length.
    codes = pd.qcut(lengths.rank(method="first"), 3, labels=False, duplicates="drop")
    work = df.assign(_bucket=codes)
    present = sorted(int(c) for c in work["_bucket"].dropna().unique())
    if len(present) < 2:
        return

    _emit(lines, "\n## 7. CER vs decode length (pooled over arms × seeds × temps)\n")
    _emit(lines, "  Compression error should grow with decode length if the long-")
    _emit(lines, "  context hypothesis holds: the gap to baseline widens left→right.\n")
    edges = work.groupby("_bucket", observed=True)["n_ar_tokens"].agg(["min", "max"])
    order = present
    labels = {
        b: f"{int(edges.loc[b, 'min'])}-{int(edges.loc[b, 'max'])}tok" for b in order
    }
    header = f"{'config':<22}" + "".join(f"{labels[b]:>20}" for b in order)
    _emit(lines, header)
    for config in work["config"].dropna().unique():
        row = f"{config:<22}"
        for b in order:
            cell = work[(work["config"] == config) & (work["_bucket"] == b)]
            cm, clo, chi = bootstrap_ci(cell["cer"], n_boot)
            row += (
                f"{f'{cm:.3f}[{clo:.3f},{chi:.3f}]':>20}"
                if not np.isnan(cm)
                else f"{'---':>20}"
            )
        _emit(lines, row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory to auto-discover the latest CSVs in (default: results).",
    )
    parser.add_argument("--trials", default=None, help="Explicit per-trial CSV path.")
    parser.add_argument(
        "--trials-glob",
        default=None,
        help="Glob of per-trial CSVs to concatenate — use this to reassemble a "
        "parallel sharded run, e.g. 'results/qwen_trials_shard*_<run-tag>.csv'.",
    )
    parser.add_argument(
        "--kv-recon", default=None, help="Explicit KV-reconstruction CSV path."
    )
    parser.add_argument(
        "--out-md", default=None, help="Also write the report to this markdown file."
    )
    parser.add_argument(
        "--n-boot", type=int, default=10000, help="Bootstrap resamples (default 10000)."
    )
    args = parser.parse_args()

    trials_df, trials_used = load_trials(
        args.trials, args.trials_glob, args.results_dir
    )
    kv_path = args.kv_recon or _latest(args.results_dir, "_kv_recon_*.csv")

    lines: list[str] = []
    _emit(lines, "# Rigorous KV-Quantization Benchmark — Analysis\n")

    if trials_df is not None:
        _emit(lines, f"Per-trial CSV(s): {', '.join(trials_used)}")
        df = _normalize_temperature(trials_df)
        section_means(df, args.n_boot, lines)
        section_paired_tests(df, lines)
        section_variance(df, lines)
        section_arm_contrast(df, lines)
        section_temperature_trend(df, args.n_boot, lines)
        section_length_trend(df, args.n_boot, lines)
    else:
        _emit(lines, "No per-trial CSV found — skipping downstream sections.")

    if kv_path and os.path.exists(kv_path):
        _emit(lines, f"\nKV-reconstruction CSV: {kv_path}")
        section_intrinsic(pd.read_csv(kv_path), lines)
    else:
        _emit(lines, "\nNo KV-reconstruction CSV found — skipping intrinsic section.")

    if args.out_md:
        with open(args.out_md, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"\nWrote markdown report to {args.out_md}")


if __name__ == "__main__":
    main()
