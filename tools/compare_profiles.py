"""Parse profile TXT dumps from the VALL-E and Qwen benchmarks, print a consolidated table.

The profile TXT files are produced by:
    python models/VALL-E-X/benchmarks/benchmark_vallex_real.py --profile ...
    python models/Qwen3-TTS/benchmarks/benchmark_qwen3tts_real.py --profile ...

Each file contains one or more "Profiling config: <name>" blocks, each with:
    Elapsed: <seconds>  Peak VRAM: <MB>
    Compressed/Realized/R_theory/R_eff (optional, TQ configs only)
    GPU-BOUND or LAUNCH-BOUND verdict line
    Top-20 by cuda_time_total table
    Top-10 by self_cpu_time_total table

Usage:
    python tools/compare_profiles.py results/profile_vallex_*.txt results/profile_qwen3tts_*.txt
    python tools/compare_profiles.py --format markdown results/*.txt > comparison.md
    python tools/compare_profiles.py --metric cuda_time results/profile_*.txt  # different sort

Also handles benchmark TXT files (per-sentence aggregates) if pointed at them.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Optional


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #

@dataclass
class ConfigProfile:
    """One (model, config) result extracted from a profile TXT."""
    source_file: str
    model: str              # "vallex" or "qwen3tts"
    config: str             # e.g. "baseline (no TQ)" or "K4/V2 rw=128"
    elapsed_s: float = 0.0
    peak_vram_mb: float = 0.0
    # Memory report (TQ configs only; zero for baseline)
    compressed_mb: float = 0.0
    realized_mb: float = 0.0
    r_theory: float = 1.0
    r_eff: float = 1.0
    # GPU-bound vs launch-bound
    verdict: str = ""
    cuda_time_frac: float = 0.0
    # Environment
    gpu_name: str = ""
    total_vram_gb: float = 0.0
    model_weight_mb: float = 0.0
    # Top kernels — list of (name, cuda_pct, calls, cuda_total_ms)
    top_cuda: list[tuple] = field(default_factory=list)
    top_cpu: list[tuple] = field(default_factory=list)


_ELAPSED_RE = re.compile(r"Elapsed:\s+([\d.]+)s\s+Peak VRAM:\s+(\d+)\s*MB")
_COMP_RE = re.compile(
    r"(?:Compressed|SimComp):\s+([\d.]+)\s*MB\s*\|\s*(?:Decomp prefix|Realized):\s+"
    r"([\d.]+)\s*MB\s*\|\s*R_theory=([\d.]+)x\s*R_eff=([\d.]+)x"
)
_COMP_RE_ALT = re.compile(
    r"Realized:\s+([\d.]+)\s*MB\s*\|\s*SimComp:\s+([\d.]+)\s*MB\s*\|\s*"
    r"R_theory=([\d.]+)x\s*R_eff=([\d.]+)x"
)
_VERDICT_RE = re.compile(r"(GPU-BOUND|LAUNCH-BOUND)\s+\(cuda_time/wall\s*=\s*([\d.]+)%\)")
_GPU_NAME_RE = re.compile(r"name=(.+)")
_TOTAL_MEM_RE = re.compile(r"total_memory=([\d.]+)\s*GB")
_WEIGHT_RE = re.compile(r"Model weights \+ load-time VRAM:\s+(\d+)\s*MB")


def _detect_model(path: str) -> str:
    base = os.path.basename(path).lower()
    if "vallex" in base or "valle" in base:
        return "vallex"
    if "qwen" in base:
        return "qwen3tts"
    return "unknown"


def _parse_top_kernels(block_lines: list[str], tag: str) -> list[tuple]:
    """Parse `prof.key_averages().table(...)` output into rows.

    Returns list of (name, cuda_pct, calls, cuda_total_ms). Very defensive —
    different torch versions format the table slightly differently.
    """
    out: list[tuple] = []
    in_table = False
    # Find the "Top-20 by cuda_time_total:" or "Top-10 by self_cpu_time_total:" section
    start_idx = None
    for i, line in enumerate(block_lines):
        if tag in line:
            start_idx = i
            break
    if start_idx is None:
        return out

    for line in block_lines[start_idx:]:
        if re.match(r"^-+\s+-+", line.strip()):
            in_table = not in_table
            continue
        if not in_table:
            continue
        if line.startswith(("Self CPU time total", "Self CUDA time total")):
            break
        # Row format (spaces-separated, columns = Name | Self CPU % | Self CPU | CPU total % |
        # CPU total | CPU time avg | Self CUDA | Self CUDA % | CUDA total | CUDA time avg | # of Calls)
        # Robust parse: split on 2+ spaces; the last numeric token is # of Calls.
        tokens = re.split(r"\s{2,}", line.strip())
        if len(tokens) < 6:
            continue
        name = tokens[0]
        # try to find CUDA % and CUDA total ms
        cuda_pct = 0.0
        cuda_ms = 0.0
        calls = 0
        try:
            calls = int(tokens[-1].replace(",", ""))
        except ValueError:
            continue
        # Last few cells: ... # of Calls  CUDA time avg  CUDA total  Self CUDA %  Self CUDA
        # In the torch printout order, Self CUDA % ends with "%" character.
        for tok in tokens:
            m = re.match(r"([\d.]+)\s*%$", tok.strip())
            if m:
                cuda_pct = max(cuda_pct, float(m.group(1)))
        # Try CUDA total column (cell containing "ms" or "us" closest to end)
        for tok in reversed(tokens):
            m = re.match(r"([\d.]+)\s*(ms|us|s)\b", tok.strip())
            if m:
                val = float(m.group(1))
                if m.group(2) == "us":
                    val /= 1000.0
                elif m.group(2) == "s":
                    val *= 1000.0
                cuda_ms = val
                break
        out.append((name, cuda_pct, calls, cuda_ms))
        if len(out) >= 20:
            break
    return out


def parse_profile_file(path: str) -> list[ConfigProfile]:
    """Split a profile TXT into per-config ConfigProfile records."""
    with open(path) as f:
        text = f.read()
    lines = text.splitlines()
    model = _detect_model(path)

    # Header: GPU config block (applies to all configs in this file).
    gpu_name = ""
    total_vram_gb = 0.0
    weight_mb = 0.0
    for line in lines[:100]:
        m = _GPU_NAME_RE.search(line)
        if m and not gpu_name:
            gpu_name = m.group(1).strip()
        m = _TOTAL_MEM_RE.search(line)
        if m and total_vram_gb == 0.0:
            total_vram_gb = float(m.group(1))
        m = _WEIGHT_RE.search(line)
        if m and weight_mb == 0.0:
            weight_mb = float(m.group(1))

    # Split into per-config blocks
    results: list[ConfigProfile] = []
    config_start = [i for i, line in enumerate(lines) if line.startswith("Profiling config:")]
    config_start.append(len(lines))  # sentinel

    for k in range(len(config_start) - 1):
        block = lines[config_start[k]:config_start[k + 1]]
        config_name = block[0].replace("Profiling config:", "").strip()
        prof = ConfigProfile(
            source_file=os.path.basename(path),
            model=model,
            config=config_name,
            gpu_name=gpu_name,
            total_vram_gb=total_vram_gb,
            model_weight_mb=weight_mb,
        )
        for line in block:
            m = _ELAPSED_RE.search(line)
            if m:
                prof.elapsed_s = float(m.group(1))
                prof.peak_vram_mb = float(m.group(2))
            m = _COMP_RE.search(line) or _COMP_RE_ALT.search(line)
            if m:
                # Both orderings are possible; match by pattern name choice.
                groups = m.groups()
                if _COMP_RE.search(line):
                    prof.compressed_mb = float(groups[0])
                    prof.realized_mb = float(groups[1])
                else:
                    prof.realized_mb = float(groups[0])
                    prof.compressed_mb = float(groups[1])
                prof.r_theory = float(groups[2])
                prof.r_eff = float(groups[3])
            m = _VERDICT_RE.search(line)
            if m:
                prof.verdict = m.group(1)
                prof.cuda_time_frac = float(m.group(2)) / 100.0
        prof.top_cuda = _parse_top_kernels(block, "Top-20 by cuda_time_total")
        prof.top_cpu = _parse_top_kernels(block, "Top-10 by self_cpu_time_total")
        results.append(prof)

    return results


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def _fmt_label(p: ConfigProfile) -> str:
    return f"{p.model}/{p.config}"


def render_headline_table(profiles: list[ConfigProfile], fmt: str = "plain") -> str:
    """Wall time, VRAM, compression ratios, verdict."""
    if not profiles:
        return "(no profiles parsed)"

    headers = [
        "Model / Config", "Wall(s)", "Peak VRAM(MB)", "Weight(MB)",
        "Dynamic(MB)", "Realized(MB)", "SimComp(MB)", "R_th", "R_eff", "Verdict",
    ]
    rows = []
    for p in profiles:
        dynamic_mb = max(p.peak_vram_mb - p.model_weight_mb, 0.0) if p.model_weight_mb else 0.0
        rows.append([
            _fmt_label(p),
            f"{p.elapsed_s:.2f}",
            f"{p.peak_vram_mb:.0f}",
            f"{p.model_weight_mb:.0f}" if p.model_weight_mb else "?",
            f"{dynamic_mb:.0f}" if p.model_weight_mb else "?",
            f"{p.realized_mb:.2f}",
            f"{p.compressed_mb:.2f}",
            f"{p.r_theory:.2f}",
            f"{p.r_eff:.2f}",
            f"{p.verdict} ({p.cuda_time_frac:.1%})" if p.verdict else "—",
        ])
    return _format_table(headers, rows, fmt, title="Headline — timing, memory, compression")


def render_top_kernels_table(profiles: list[ConfigProfile], top_n: int = 5, fmt: str = "plain") -> str:
    if not profiles:
        return ""
    headers = ["Model / Config"] + [f"Top-{i+1} (CUDA %)" for i in range(top_n)]
    rows = []
    for p in profiles:
        row = [_fmt_label(p)]
        for i in range(top_n):
            if i < len(p.top_cuda):
                name, pct, calls, _ = p.top_cuda[i]
                row.append(f"{name} ({pct:.1f}%, {calls}x)")
            else:
                row.append("—")
        rows.append(row)
    return _format_table(headers, rows, fmt, title="Top kernels by CUDA time")


def _format_table(headers: list[str], rows: list[list[str]], fmt: str, title: str = "") -> str:
    if fmt == "markdown":
        out = [f"\n### {title}\n"] if title else []
        out.append("| " + " | ".join(headers) + " |")
        out.append("|" + "|".join(["---"] * len(headers)) + "|")
        for r in rows:
            out.append("| " + " | ".join(str(x) for x in r) + " |")
        return "\n".join(out)
    # plain — fixed-width
    widths = [max(len(headers[i]), max((len(str(r[i])) for r in rows), default=0)) for i in range(len(headers))]
    out = []
    if title:
        out.append(f"\n=== {title} ===")
    out.append("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    out.append("  ".join("-" * widths[i] for i in range(len(headers))))
    for r in rows:
        out.append("  ".join(str(r[i]).ljust(widths[i]) for i in range(len(headers))))
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Compare profile TXT dumps from VALL-E and Qwen benchmarks.",
    )
    p.add_argument("files", nargs="+", help="One or more profile_*.txt paths (globs expanded).")
    p.add_argument("--format", choices=["plain", "markdown"], default="plain")
    p.add_argument("--top-n", type=int, default=5, help="Top-N kernels per config (default 5).")
    args = p.parse_args(argv)

    paths: list[str] = []
    for f in args.files:
        expanded = glob.glob(f) or [f]
        paths.extend(expanded)
    paths = sorted(set(paths))

    all_profiles: list[ConfigProfile] = []
    for path in paths:
        if not os.path.exists(path):
            print(f"WARNING: not found: {path}", file=sys.stderr)
            continue
        try:
            all_profiles.extend(parse_profile_file(path))
        except Exception as e:
            print(f"WARNING: failed to parse {path}: {e}", file=sys.stderr)

    if not all_profiles:
        print("No profiles parsed. Pass paths to profile_*.txt files.", file=sys.stderr)
        return 2

    # Sort so baseline comes first per model.
    def _sort_key(cp: ConfigProfile):
        return (cp.model, 0 if "baseline" in cp.config else 1, cp.config)

    all_profiles.sort(key=_sort_key)

    print(render_headline_table(all_profiles, fmt=args.format))
    print(render_top_kernels_table(all_profiles, top_n=args.top_n, fmt=args.format))

    # A bit of bonus context: environment summary
    environments = {(p.gpu_name, p.total_vram_gb) for p in all_profiles if p.gpu_name}
    if environments:
        print()
        if args.format == "markdown":
            print("### Environment\n")
        else:
            print("=== Environment ===")
        for name, vram in environments:
            print(f"- {name} ({vram:.1f} GB)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
