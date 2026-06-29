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

from turboquant.bench_common import degradation_onset  # noqa: E402  (used in §windowed)


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


def _emit_table(
    lines: list[str],
    headers: list[str],
    rows: list[tuple],
    aligns: Optional[list[str]] = None,
) -> None:
    """Emit a GitHub-markdown table, cells padded so it also aligns in the console.

    ``aligns`` is per-column "<" (left) or ">" (right); default = first column
    left, the rest right. Each cell is pre-formatted by the caller (this only lays
    out the grid), so numeric precision is unchanged from the plain-text version.
    """
    cells = [[str(c) for c in headers]] + [[str(c) for c in r] for r in rows]
    widths = [max(len(row[i]) for row in cells) for i in range(len(headers))]
    aligns = aligns or (["<"] + [">"] * (len(headers) - 1))

    def _fmt(row: list[str]) -> str:
        out = [
            c.ljust(w) if a == "<" else c.rjust(w)
            for c, w, a in zip(row, widths, aligns)
        ]
        return "| " + " | ".join(out) + " |"

    sep = [
        (":" + "-" * max(1, w - 1)) if a == "<" else ("-" * max(1, w - 1) + ":")
        for w, a in zip(widths, aligns)
    ]
    _emit(lines, _fmt(cells[0]))
    _emit(lines, "| " + " | ".join(sep) + " |")
    for row in cells[1:]:
        _emit(lines, _fmt(row))


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
        rows = []
        for group in sorted(sub["group"].dropna().unique()):
            for temp in sorted(sub[sub["group"] == group]["temperature"].unique()):
                cells = sub[(sub["group"] == group) & (sub["temperature"] == temp)]
                for config in cells["config"].unique():
                    cell = cells[cells["config"] == config]
                    cm, clo, chi = bootstrap_ci(cell["cer"], n_boot)
                    sm, _, _ = bootstrap_ci(cell["spk_sim"], n_boot)
                    smr, _, _ = bootstrap_ci(_col(cell, "spk_sim_ref"), n_boot)
                    sm_str = "---" if np.isnan(sm) else f"{sm:.4f}"
                    smr_str = "---" if np.isnan(smr) else f"{smr:.4f}"
                    rows.append(
                        (
                            group,
                            str(temp),
                            config,
                            len(cell),
                            f"{cm:.4f}",
                            f"[{clo:.4f}, {chi:.4f}]",
                            sm_str,
                            smr_str,
                        )
                    )
        _emit_table(
            lines,
            [
                "group",
                "temp",
                "config",
                "n",
                "CER mean",
                "[95% CI]",
                "SpkSim",
                "SpkSimRef",
            ],
            rows,
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
        table = []
        for (c, n, md, p), pa in zip(rows, adj):
            md_str = "n/a" if np.isnan(md) else f"{md:+.4f}"
            p_str = "n/a" if np.isnan(p) else f"{p:.4f}"
            table.append((c, n, md_str, p_str, f"{pa:.4f}", sig_marker(pa)))
        _emit_table(
            lines,
            ["config", "n_pairs", "median ΔCER", "p_raw", "p_holm", "sig"],
            table,
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
    rows = [
        (
            r["config"],
            r["variant"],
            f"{r['cos_k']:.4f}",
            f"{r['cos_v']:.4f}",
            f"{r['relmse_k']:.4g}",
            f"{r['relmse_v']:.4g}",
        )
        for _, r in g.iterrows()
    ]
    _emit_table(
        lines,
        ["config", "variant", "cos_k", "cos_v", "relmse_k", "relmse_v"],
        rows,
        aligns=["<", "<", ">", ">", ">", ">"],
    )
    _emit(lines)


def section_arm_contrast(df: pd.DataFrame, lines: list[str]) -> None:
    arms = set(df["arm"].dropna().unique())
    if not {"greedy", "sampling"}.issubset(arms):
        return
    _emit(lines, "\n## 5. Greedy vs sampling contrast (mean CER by config)\n")
    pivot = df.pivot_table(index="config", columns="arm", values="cer", aggfunc="mean")
    rows = []
    for config, row in pivot.iterrows():
        gd = row.get("greedy", np.nan)
        sp = row.get("sampling", np.nan)
        rows.append((config, f"{gd:.4f}", f"{sp:.4f}", f"{sp - gd:.4f}"))
    _emit_table(lines, ["config", "greedy", "sampling", "Δ(samp-greedy)"], rows)
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
    rows = []
    for config in samp["config"].unique():
        cells = [config]
        for t in temps:
            cell = samp[(samp["config"] == config) & (samp["temperature"] == t)]
            cm, clo, chi = bootstrap_ci(cell["cer"], n_boot)
            cells.append("---" if np.isnan(cm) else f"{cm:.3f}[{clo:.3f},{chi:.3f}]")
        rows.append(tuple(cells))
    _emit_table(lines, ["config", *(f"T={t}" for t in temps)], rows)


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
    rows = []
    for config in work["config"].dropna().unique():
        cells = [config]
        for b in order:
            cell = work[(work["config"] == config) & (work["_bucket"] == b)]
            cm, clo, chi = bootstrap_ci(cell["cer"], n_boot)
            cells.append("---" if np.isnan(cm) else f"{cm:.3f}[{clo:.3f},{chi:.3f}]")
        rows.append(tuple(cells))
    _emit_table(lines, ["config", *(labels[b] for b in order)], rows)


_LENGTHS = ("short", "medium", "long")
_DIFFS = ("easy", "medium", "hard")


def _parse_cell(group: str) -> tuple[Optional[str], Optional[str]]:
    """Split a grid-cell group ``"<length>_<difficulty>"`` → (length, difficulty).

    Returns (None, None) for non-grid groups (smoke/long/seedtts_en/…), so the
    section silently skips them.
    """
    parts = str(group).split("_", 1)
    if len(parts) == 2 and parts[0] in _LENGTHS and parts[1] in _DIFFS:
        return parts[0], parts[1]
    return None, None


def attach_ref_quality(df: pd.DataFrame, path: Optional[str]) -> pd.DataFrame:
    """Left-join the ref_quality sidecar (sentence_hash → clean/ref_cer/ref_snr).

    No-op (returns df unchanged) when no path/file or no sentence_hash column, so
    the analysis still runs on CSVs produced before reference tagging existed.
    """
    if not path or not os.path.exists(path) or "sentence_hash" not in df.columns:
        return df
    rq = pd.read_csv(path)[["sentence_hash", "clean", "ref_cer", "ref_snr_db"]]
    rq = rq.drop_duplicates("sentence_hash")
    return df.merge(rq, on="sentence_hash", how="left")


def section_length_difficulty(df: pd.DataFrame, n_boot: int, lines: list[str]) -> None:
    """The 3×3 experiment grid: CER-vs-length curve + per-config length×difficulty.

    Emitted only when the run uses the ``<length>_<difficulty>`` grid cells. The
    headline (difficulty pooled) is the rw=0 degradation curve — short→long at a
    fixed bit-width — which the length-gated Exp-1 could not produce.
    """
    cells = df["group"].dropna().map(lambda g: _parse_cell(g)[0] is not None)
    grid = df[cells.reindex(df.index, fill_value=False)].copy()
    if grid.empty:
        return
    grid["length"] = grid["group"].map(lambda g: _parse_cell(g)[0])
    grid["difficulty"] = grid["group"].map(lambda g: _parse_cell(g)[1])
    lengths = [le for le in _LENGTHS if le in set(grid["length"])]
    diffs = [d for d in _DIFFS if d in set(grid["difficulty"])]
    configs = list(grid["config"].dropna().unique())

    _emit(lines, "\n## 8. Length × difficulty grid (CER)\n")
    _emit(lines, "  Headline: CER vs length per config (difficulty pooled) — the")
    _emit(lines, "  degradation curve. Gap to baseline should widen short→long.\n")
    curve = []
    for config in configs:
        cells = [config]
        for le in lengths:
            cell = grid[(grid["config"] == config) & (grid["length"] == le)]
            cm, clo, chi = bootstrap_ci(cell["cer"], n_boot)
            cells.append("---" if np.isnan(cm) else f"{cm:.3f}[{clo:.3f},{chi:.3f}]")
        curve.append(tuple(cells))
    _emit_table(lines, ["config", *lengths], curve)

    _emit(lines, "\n  Per-config CER by length (rows) × difficulty (cols):")
    for config in configs:
        _emit(lines, f"\n### {config}")
        rows = []
        for le in lengths:
            cells = [le]
            for d in diffs:
                cell = grid[
                    (grid["config"] == config)
                    & (grid["length"] == le)
                    & (grid["difficulty"] == d)
                ]
                cm, _, _ = bootstrap_ci(cell["cer"], n_boot)
                cells.append("---" if np.isnan(cm) else f"{cm:.3f}")
            rows.append(tuple(cells))
        _emit_table(lines, ["length", *diffs], rows)


def section_clean_subset(df: pd.DataFrame, n_boot: int, lines: list[str]) -> None:
    """Clone-fidelity SpkSimRef per config on the clean-reference subset vs all.

    Needs the ref_quality join (a ``clean`` column). Noisy references depress
    absolute SpkSimRef, so reporting the clean subset alongside all-rows shows the
    compression drift uncontaminated by reference quality.
    """
    if "clean" not in df.columns or "spk_sim_ref" not in df.columns:
        return
    clean = df[df["clean"] == 1]
    if clean.empty:
        return
    _emit(lines, "\n## 9. SpkSimRef: clean-reference subset vs all\n")
    n_clean = (
        clean["sentence_hash"].nunique() if "sentence_hash" in clean else len(clean)
    )
    n_all = df["sentence_hash"].nunique() if "sentence_hash" in df else len(df)
    _emit(lines, f"  clean refs: {n_clean} sentences (of {n_all}).\n")
    rows = []
    for config in df["config"].dropna().unique():
        a, _, _ = bootstrap_ci(_col(df[df["config"] == config], "spk_sim_ref"), n_boot)
        c, _, _ = bootstrap_ci(
            _col(clean[clean["config"] == config], "spk_sim_ref"), n_boot
        )
        a_str = "---" if np.isnan(a) else f"{a:.4f}"
        c_str = "---" if np.isnan(c) else f"{c:.4f}"
        rows.append((config, a_str, c_str))
    _emit_table(lines, ["config", "SpkSimRef(all)", "SpkSimRef(clean)"], rows)


_LENGTH_EDGES = [(0, 128), (128, 512), (512, 1024), (1024, 2048), (2048, None)]


def _length_bucket(tok: float):
    for lo, hi in _LENGTH_EDGES:
        if tok >= lo and (hi is None or tok < hi):
            return (lo, hi)
    return None


def _bucket_label(b: tuple) -> str:
    lo, hi = b
    return f"{lo}-{hi}tok" if hi is not None else f"{lo}+tok"


def section_length_sweep_fixed(df: pd.DataFrame, n_boot: int, lines: list[str]) -> None:
    """Headline degradation curve: CER + SpkSim vs FIXED decode-length buckets.

    Unlike the rank-tercile ``section_length_trend``, the buckets are absolute
    talker-token ranges (≤128 / 128–512 / 512–1024 / 1024–2048 / 2048+), so the
    curve reads directly as 'quality vs sequence length' across the standard sets.
    """
    if "n_ar_tokens" not in df.columns:
        return
    tok = pd.to_numeric(df["n_ar_tokens"], errors="coerce")
    work = df.assign(_tok=tok).dropna(subset=["_tok"])
    if work.empty:
        return
    work = work.assign(_b=work["_tok"].map(_length_bucket))
    buckets = [b for b in _LENGTH_EDGES if (work["_b"] == b).any()]
    if len(buckets) < 2:
        return

    _emit(lines, "\n## Length sweep — CER & SpkSim vs fixed decode-length bucket\n")
    for metric, title in (("cer", "CER"), ("spk_sim", "SpkSim")):
        if metric not in work.columns:
            continue
        _emit(lines, f"### {title}")
        rows = []
        for config in work["config"].dropna().unique():
            cells = [config]
            for b in buckets:
                cell = work[(work["config"] == config) & (work["_b"] == b)]
                m, lo, hi = bootstrap_ci(cell[metric], n_boot)
                cells.append("---" if np.isnan(m) else f"{m:.3f}[{lo:.3f},{hi:.3f}]")
            rows.append(tuple(cells))
        _emit_table(lines, ["config", *(_bucket_label(b) for b in buckets)], rows)
        _emit(lines)


def _windowed_baseline(configs: list[str]) -> Optional[str]:
    for c in configs:
        if BASELINE_HINT in str(c).lower():
            return c
    return None


def section_windowed(windowed_glob: Optional[str], lines: list[str]) -> None:
    """CER(t)/SpkSim(t) degradation curves vs baseline + degradation-onset.

    Reads the windowed sidecar (one row per trial×window), averages each metric
    over items/seeds per (config, window_idx), then locates the first window where
    a compressed config peels away from baseline (SpkSim drop / CER rise).
    """
    if not windowed_glob:
        return
    paths = sorted(glob.glob(windowed_glob))
    if not paths:
        return
    w = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    if w.empty:
        return
    configs = list(w["config"].dropna().unique())
    base = _windowed_baseline(configs)
    if base is None:
        _emit(lines, "\n## Windowed degradation — (no baseline config found)\n")
        return

    sim = w.groupby(["config", "window_idx"])["spk_sim_win"].mean().unstack(0)
    cer = w.groupby(["config", "window_idx"])["cer_win"].mean().unstack(0)
    t_by_win = w.groupby("window_idx")["t_start"].mean()

    _emit(
        lines, "\n## Windowed temporal degradation (long audio) — onset vs baseline\n"
    )
    _emit(lines, "  First window where a config peels from baseline: SpkSim drop")
    _emit(lines, "  > 0.10 or CER rise > 0.20. '~t' is that window's mean start (s).\n")
    onset_rows = []
    for cfg in configs:
        if cfg == base:
            continue
        sim_on = (
            degradation_onset(sim[base].tolist(), sim[cfg].tolist(), True, 0.10)
            if cfg in sim
            else None
        )
        cer_on = (
            degradation_onset(cer[base].tolist(), cer[cfg].tolist(), False, 0.20)
            if cfg in cer
            else None
        )

        def _fmt(on):
            if on is None:
                return "none", "--"
            return f"win{on}", f"{t_by_win.get(on, float('nan')):.1f}"

        s_on, s_t = _fmt(sim_on)
        c_on, c_t = _fmt(cer_on)
        onset_rows.append((cfg, s_on, s_t, c_on, c_t))
    _emit_table(
        lines,
        ["config", "SIM onset", "~t(s)", "CER onset", "~t(s)"],
        onset_rows,
    )

    # SpkSim(t) & CER(t) vs ABSOLUTE TIME, binned across the whole duration so the
    # decline over audio length is visible regardless of how many windows each clip
    # has. Later bins are reached only by the longer clips (that's the point).
    tmax = float(w["t_start"].max())
    nbins = 12
    if tmax <= 0:
        return
    edges = np.linspace(0.0, tmax + 1e-6, nbins + 1)
    centers = [(edges[i] + edges[i + 1]) / 2 for i in range(nbins)]
    wb = w.assign(
        _tb=pd.cut(w["t_start"], bins=edges, labels=False, include_lowest=True)
    )
    for metric, title in (("spk_sim_win", "SpkSim(t)"), ("cer_win", "CER(t)")):
        piv = wb.groupby(["config", "_tb"])[metric].mean().unstack(0)
        present = [b for b in range(nbins) if b in piv.index]
        if not present:
            continue
        _emit(lines, f"\n  {title} vs audio time (mean over items), bins in seconds:")
        rows = []
        for cfg in configs:
            if cfg not in piv.columns:
                continue
            cells = [cfg]
            for b in present:
                v = piv.loc[b, cfg] if b in piv.index else float("nan")
                cells.append("---" if (v != v) else f"{v:.2f}")
            rows.append(tuple(cells))
        _emit_table(lines, ["config", *(f"{centers[b]:.0f}s" for b in present)], rows)


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
        "--ref-quality",
        default=None,
        help="ref_quality.csv from validate_eval_set.py (sentence_hash → clean / "
        "ref_cer / ref_snr). Enables the clean-reference-subset SpkSimRef section.",
    )
    parser.add_argument(
        "--windowed",
        default=None,
        help="Glob of windowed-metrics sidecar CSVs (qwen_windowed_shard*_<tag>.csv). "
        "Enables the CER(t)/SpkSim(t) degradation-curve + onset section.",
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
        df = attach_ref_quality(df, args.ref_quality)
        section_means(df, args.n_boot, lines)
        section_paired_tests(df, lines)
        section_variance(df, lines)
        section_arm_contrast(df, lines)
        section_temperature_trend(df, args.n_boot, lines)
        section_length_trend(df, args.n_boot, lines)
        section_length_sweep_fixed(df, args.n_boot, lines)
        section_length_difficulty(df, args.n_boot, lines)
        section_clean_subset(df, args.n_boot, lines)
    else:
        _emit(lines, "No per-trial CSV found — skipping downstream sections.")

    section_windowed(args.windowed, lines)

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
