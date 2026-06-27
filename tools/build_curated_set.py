"""Assemble the curated 3×3 eval cells from raw candidate pools — reproducibly.

The curated cells (medium/long lengths + all 'hard' difficulties) are
LLM-generated text, NOT a standard set. This script is the programmatic gate
that turns oversized raw candidate pools into the final per-cell files:

  data/curated/_candidates/<cell>.txt   (raw, oversized, one sentence per line)
        ->  filter by predict_tokens length band  (length matches the cell)
        ->  filter by text_difficulty class        (difficulty matches the cell)
        ->  case-insensitive dedup, drop blanks/comments
        ->  keep the first N (default 100)
  data/curated/<cell>.txt               (final, with a provenance header)

Length/difficulty are judged by the SAME heuristics the loader and analysis use
(turboquant.eval_sentences), so "what the set claims" and "what gets loaded"
cannot drift. Baseline-CER validation (easy<medium<hard, drop un-synthesizable)
is a separate GPU step: tools/validate_eval_set.py.

Run: python tools/build_curated_set.py [--n 100] [--data-dir data]
"""

from __future__ import annotations

import argparse
import os

from turboquant.eval_sentences import (
    _CURATED_CELLS,
    length_category,
    text_difficulty,
)


def _read_candidates(path: str) -> list[str]:
    if not os.path.exists(path):
        return []
    out: list[str] = []
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if line and not line.startswith("#"):
                out.append(line)
    return out


def filter_cell(candidates: list[str], cell: str, n: int) -> tuple[list[str], dict]:
    """Keep the first ``n`` candidates whose length band + difficulty match ``cell``.

    Returns (kept, stats). ``stats`` reports why candidates were dropped so a
    short pool is diagnosable (wrong length vs wrong difficulty vs duplicate).
    """
    want_length, want_diff = cell.split("_", 1)
    kept: list[str] = []
    seen: set[str] = set()
    stats = {"total": len(candidates), "bad_length": 0, "bad_diff": 0, "dup": 0}
    for text in candidates:
        key = " ".join(text.lower().split())
        if key in seen:
            stats["dup"] += 1
            continue
        if length_category(text) != want_length:
            stats["bad_length"] += 1
            continue
        if text_difficulty(text) != want_diff:
            stats["bad_diff"] += 1
            continue
        seen.add(key)
        kept.append(text)
        if len(kept) >= n:
            break
    stats["kept"] = len(kept)
    return kept, stats


def _provenance_header(cell: str, n_kept: int) -> str:
    length, diff = cell.split("_", 1)
    lo, hi = {
        "short": (32, 64),
        "medium": (128, 256),
        "long": (512, 1024),
    }[length]
    return (
        f"# Curated eval cell: {cell}  ({n_kept} sentences)\n"
        f"# LENGTH={length} (target ~{lo}-{hi} talker tokens, ≈6.4×words),"
        f" DIFFICULTY={diff}.\n"
        f"# GENERATED text (not a standard set) — filtered by predict_tokens band\n"
        f"# + text_difficulty class via tools/build_curated_set.py. Validate baseline\n"
        f"# CER ordering with tools/validate_eval_set.py before citing.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--n", type=int, default=100, help="Target per cell.")
    parser.add_argument(
        "--candidates-dir",
        default=None,
        help="Raw candidate pools (default: <data-dir>/curated/_candidates).",
    )
    args = parser.parse_args()

    cand_dir = args.candidates_dir or os.path.join(
        args.data_dir, "curated", "_candidates"
    )
    out_dir = os.path.join(args.data_dir, "curated")
    os.makedirs(out_dir, exist_ok=True)

    print(
        f"{'cell':<16} {'total':>6} {'kept':>6} {'badLen':>7} {'badDiff':>8} {'dup':>5}"
    )
    short = []
    for cell in _CURATED_CELLS:
        candidates = _read_candidates(os.path.join(cand_dir, f"{cell}.txt"))
        kept, stats = filter_cell(candidates, cell, args.n)
        with open(os.path.join(out_dir, f"{cell}.txt"), "w", encoding="utf-8") as fh:
            fh.write(_provenance_header(cell, len(kept)))
            fh.write("\n".join(kept) + ("\n" if kept else ""))
        print(
            f"{cell:<16} {stats['total']:>6} {stats['kept']:>6} "
            f"{stats['bad_length']:>7} {stats['bad_diff']:>8} {stats['dup']:>5}"
        )
        if len(kept) < args.n:
            short.append((cell, len(kept)))

    if short:
        print("\nUNDER TARGET (add more candidates and re-run):")
        for cell, k in short:
            print(f"  {cell}: {k}/{args.n}")
    else:
        print(f"\nAll {len(_CURATED_CELLS)} curated cells at {args.n}.")


if __name__ == "__main__":
    main()
