"""Analyze benchmark log files and produce summary tables.

Usage:
    python analyze_results.py results/benchmark_20260402_214427.log
    python analyze_results.py results/  # analyze all logs in directory
"""

import re
import sys
import os
import json
from collections import defaultdict


def parse_log(filepath):
    """Parse a benchmark log file into structured results."""
    with open(filepath) as f:
        text = f.read()

    results = {
        "generation": {},       # group -> config -> {rtf: [], cer: [], spk_sim: []}
        "averages": {},         # group -> config -> {rtf, cer, spk_sim}
        "attention_similarity": {},  # config -> {short, medium, long}
    }

    # Parse per-sentence results: "config   RTF=X.XX CER=X.X% SpkSim=X.XXXX"
    current_group = None
    group_pattern = re.compile(r"Group: (\w+) \((\d+) sentences\)")
    result_pattern = re.compile(
        r"^\s+(.+?)\s{2,}RTF=([\d.]+)"
        r"(?:\s+CER=([\d.]+)%)?"
        r"(?:\s+SpkSim=([\d.]+))?",
    )

    for line in text.split("\n"):
        gm = group_pattern.search(line)
        if gm:
            current_group = gm.group(1)
            if current_group not in results["generation"]:
                results["generation"][current_group] = defaultdict(
                    lambda: {"rtf": [], "cer": [], "spk_sim": []}
                )
            continue

        if current_group:
            rm = result_pattern.match(line)
            if rm:
                config = rm.group(1).strip()
                rtf = float(rm.group(2))
                cer = float(rm.group(3)) / 100 if rm.group(3) else None
                spk_sim = float(rm.group(4)) if rm.group(4) else None

                results["generation"][current_group][config]["rtf"].append(rtf)
                if cer is not None:
                    results["generation"][current_group][config]["cer"].append(cer)
                if spk_sim is not None:
                    results["generation"][current_group][config]["spk_sim"].append(spk_sim)

    # Parse AVERAGES blocks
    # Lines look like: "  K4/V2 rw=128           6.43       0.29%      0.9417"
    # or:              "  baseline (no TQ)       1.73       0.65%      ---"
    avg_pattern = re.compile(
        r"^\s+(.+?)\s{2,}([\d.]+)\s+([\d.]+)%\s+([\d.]+|---)"
    )
    in_averages = False
    avg_group = None

    for line in text.split("\n"):
        if "AVERAGES for" in line:
            m = re.search(r"AVERAGES for (\w+)", line)
            if m:
                avg_group = m.group(1)
                results["averages"][avg_group] = {}
                in_averages = True
            continue

        if in_averages and avg_group:
            am = avg_pattern.match(line)
            if am:
                config = am.group(1).strip()
                if config in ("Config", "---"):
                    continue
                results["averages"][avg_group][config] = {
                    "rtf": float(am.group(2)),
                    "cer": float(am.group(3)) / 100,
                    "spk_sim": float(am.group(4)) if am.group(4) != "---" else None,
                }
            elif line.strip() == "" or "===" in line:
                in_averages = False
                avg_group = None

    # Parse Attention Similarity Summary
    attn_pattern = re.compile(r"^\s+(K\d/V\d)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)")
    for line in text.split("\n"):
        am = attn_pattern.match(line)
        if am:
            config = am.group(1)
            results["attention_similarity"][config] = {
                "short": float(am.group(2)),
                "medium": float(am.group(3)),
                "long": float(am.group(4)),
            }

    return results


def print_summary(results):
    """Print a clean summary of parsed results."""
    groups = list(results["averages"].keys())
    if not groups:
        print("No averaged results found in log.")
        return

    configs = list(results["averages"][groups[0]].keys())

    # --- Generation Summary ---
    print("=" * 80)
    print("GENERATION RESULTS (averaged)")
    print("=" * 80)

    for group in groups:
        n = 0
        for config in configs:
            gen = results["generation"].get(group, {}).get(config, {})
            n = max(n, len(gen.get("rtf", [])))

        print(f"\n  {group} ({n} sentences):")
        print(f"  {'Config':<22} {'RTF':<8} {'CER':<8} {'SpkSim':<10} {'RTF vs BL':<10}")
        print(f"  {'-' * 58}")

        baseline_rtf = results["averages"][group].get(configs[0], {}).get("rtf", 1)

        for config in configs:
            r = results["averages"][group].get(config, {})
            rtf = r.get("rtf", 0)
            cer = r.get("cer", 0)
            spk_sim = r.get("spk_sim")
            spk_str = f"{spk_sim:.4f}" if spk_sim else "---"
            overhead = f"{rtf / baseline_rtf:.1f}x" if baseline_rtf > 0 else "---"
            print(f"  {config:<22} {rtf:<8.2f} {cer:<8.2%} {spk_str:<10} {overhead:<10}")

    # --- Attention Similarity ---
    if results["attention_similarity"]:
        print(f"\n{'=' * 80}")
        print("ATTENTION SIMILARITY (KV reconstruction cosine similarity)")
        print("=" * 80)
        print(f"  {'Config':<10}", end="")
        for group in ["short", "medium", "long"]:
            print(f" {group:<10}", end="")
        print(f" {'avg':<10}")
        print(f"  {'-' * 50}")

        for config, sims in results["attention_similarity"].items():
            vals = [sims.get(g, 0) for g in ["short", "medium", "long"]]
            avg = sum(vals) / len(vals) if vals else 0
            print(f"  {config:<10}", end="")
            for v in vals:
                print(f" {v:<10.4f}", end="")
            print(f" {avg:<10.4f}")

    # --- Key Insights ---
    print(f"\n{'=' * 80}")
    print("KEY INSIGHTS")
    print("=" * 80)

    # Best config by CER
    for group in groups:
        best_config = None
        best_cer = float("inf")
        for config in configs:
            r = results["averages"][group].get(config, {})
            cer = r.get("cer", 1)
            if "baseline" not in config and cer < best_cer:
                best_cer = cer
                best_config = config
        baseline_cer = results["averages"][group].get(configs[0], {}).get("cer", 0)
        if best_config:
            print(f"  {group}: best CER = {best_config} ({best_cer:.2%}) vs baseline ({baseline_cer:.2%})")

    # Broken configs (CER > 20%)
    broken = []
    for group in groups:
        for config in configs:
            cer = results["averages"][group].get(config, {}).get("cer", 0)
            if cer > 0.2 and "baseline" not in config:
                broken.append(f"{config} on {group} ({cer:.1%})")
    if broken:
        print(f"\n  BROKEN configs (CER > 20%): {', '.join(broken)}")

    # Best attention similarity
    if results["attention_similarity"]:
        best_attn = max(
            results["attention_similarity"].items(),
            key=lambda x: sum(x[1].values()) / len(x[1]),
        )
        avg_sim = sum(best_attn[1].values()) / len(best_attn[1])
        print(f"\n  Best attention similarity: {best_attn[0]} ({avg_sim:.4f})")


def export_json(results, filepath):
    """Export results to JSON for further analysis."""
    # Convert defaultdicts to regular dicts
    out = {
        "averages": results["averages"],
        "attention_similarity": results["attention_similarity"],
        "generation": {},
    }
    for group, configs in results["generation"].items():
        out["generation"][group] = {}
        for config, data in configs.items():
            out["generation"][group][config] = dict(data)

    json_path = filepath.replace(".log", ".json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nExported to: {json_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <log_file_or_directory>")
        sys.exit(1)

    path = sys.argv[1]

    if os.path.isdir(path):
        logs = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith(".log")])
    else:
        logs = [path]

    for log in logs:
        print(f"\n{'#' * 80}")
        print(f"# Analyzing: {log}")
        print(f"{'#' * 80}")

        results = parse_log(log)
        print_summary(results)
        export_json(results, log)


if __name__ == "__main__":
    main()
