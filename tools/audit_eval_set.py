"""Audit the evaluation set BEFORE the (long) GPU run — no model, no GPU.

Loads exactly what the benchmark will load (same ``iter_eval_items`` +
``--max-per-group``) and reports, per group:

  - sentence count (after the cap) and word/predicted-token distribution,
  - length-bucket coverage (≤128 / 128–512 / 512–1024 / 1024–2048+ talker tokens)
    so the length sweep is guaranteed data in each bucket,
  - reference-audio coverage for cloning: how many items carry a ref, and (with
    ``--check-audio``) how many of those files actually exist on disk,
  - ground-truth-audio coverage (for SpkSimRef vs a real clip).

Then a run-level summary: total sentences, projected trial count
(sentences × configs × seeds × temps), and explicit WARNINGS for anything that
would make the run skip cells (clone mode with missing refs, empty groups,
length buckets with no coverage).

Run on the box after fetch/build:
  python tools/audit_eval_set.py --data-dir data \
      --groups seedtts_en,librispeech_pc,libritts_long,ellav_hard \
      --max-per-group 100 --voice-mode clone --n-configs 5 --check-audio
"""

from __future__ import annotations

import argparse
import os
import statistics

from turboquant.eval_sentences import (
    available_groups,
    iter_eval_items,
    predict_tokens,
)

_BUCKETS = [(0, 128), (128, 512), (512, 1024), (1024, 2048), (2048, None)]


def _bucket_label(lo: int, hi) -> str:
    return f"{lo}-{hi}" if hi is not None else f"{lo}+"


def _bucket_of(tok: int) -> str:
    for lo, hi in _BUCKETS:
        if tok >= lo and (hi is None or tok < hi):
            return _bucket_label(lo, hi)
    return "?"


def _exists(path) -> bool:
    return bool(path) and os.path.exists(path)


def audit_group(group, items, voice_mode, check_audio):
    """Return a stats dict + a list of warning strings for one group."""
    warns = []
    n = len(items)
    if n == 0:
        warns.append(f"{group}: 0 sentences loaded (group empty or data missing)")
        return {"group": group, "n": 0}, warns

    toks = [predict_tokens(it.text) for it in items]
    words = [len(it.text.split()) for it in items]
    lens = {_bucket_label(lo, hi): 0 for lo, hi in _BUCKETS}
    for t in toks:
        lens[_bucket_of(t)] += 1

    with_ref = [it for it in items if it.ref_audio]
    with_gt = [it for it in items if it.ground_truth_audio]
    ref_missing_disk = (
        sum(1 for it in with_ref if not _exists(it.ref_audio)) if check_audio else None
    )
    gt_missing_disk = (
        sum(1 for it in with_gt if not _exists(it.ground_truth_audio))
        if check_audio
        else None
    )

    # Clone mode needs a reference for every item (own or default).
    if voice_mode == "clone":
        no_ref = n - len(with_ref)
        if no_ref and group in ("seedtts_en", "librispeech_pc", "libritts_long"):
            warns.append(
                f"{group}: {no_ref}/{n} items have NO own ref_audio — they will need "
                f"--default-ref-audio or they get skipped in clone mode"
            )
    if check_audio and ref_missing_disk:
        warns.append(
            f"{group}: {ref_missing_disk}/{len(with_ref)} ref_audio files MISSING on "
            f"disk → those cells will skip clone (fetch incomplete?)"
        )

    stats = {
        "group": group,
        "n": n,
        "words_min": min(words),
        "words_med": int(statistics.median(words)),
        "words_max": max(words),
        "tok_min": min(toks),
        "tok_med": int(statistics.median(toks)),
        "tok_max": max(toks),
        "buckets": lens,
        "with_ref": len(with_ref),
        "with_gt": len(with_gt),
        "ref_missing_disk": ref_missing_disk,
        "gt_missing_disk": gt_missing_disk,
    }
    return stats, warns


def render(per_group, args, lines):
    lines.append("# Eval-set audit (pre-run, no GPU)\n")
    lines.append(
        f"data-dir={args.data_dir}  voice-mode={args.voice_mode}  "
        f"max-per-group={args.max_per_group}\n"
    )

    # Per-group table.
    lines.append(
        f"{'group':<16}{'n':>5}{'words(min/med/max)':>20}"
        f"{'tok(min/med/max)':>20}{'ref':>6}{'gt':>5}"
    )
    lines.append("-" * 72)
    total = 0
    for s in per_group:
        if s["n"] == 0:
            lines.append(f"{s['group']:<16}{0:>5}   (empty)")
            continue
        total += s["n"]
        wmm = f"{s['words_min']}/{s['words_med']}/{s['words_max']}"
        tmm = f"{s['tok_min']}/{s['tok_med']}/{s['tok_max']}"
        ref = s["with_ref"]
        if s["ref_missing_disk"]:
            ref = f"{ref}(-{s['ref_missing_disk']})"
        lines.append(
            f"{s['group']:<16}{s['n']:>5}{wmm:>20}{tmm:>20}{str(ref):>6}{s['with_gt']:>5}"
        )

    # Length-bucket coverage (pooled).
    lines.append("\n## Length-bucket coverage (predicted talker tokens)\n")
    labels = [_bucket_label(lo, hi) for lo, hi in _BUCKETS]
    header = f"{'group':<16}" + "".join(f"{lb:>12}" for lb in labels)
    lines.append(header)
    pooled = {lb: 0 for lb in labels}
    for s in per_group:
        if s["n"] == 0:
            continue
        row = f"{s['group']:<16}"
        for lb in labels:
            c = s["buckets"][lb]
            pooled[lb] += c
            row += f"{c:>12}"
        lines.append(row)
    lines.append(f"{'ALL':<16}" + "".join(f"{pooled[lb]:>12}" for lb in labels))

    # Run-level projection.
    trials = total * args.n_configs * args.seeds * args.temps
    lines.append("\n## Run projection\n")
    lines.append(f"  sentences (all groups, after cap): {total}")
    lines.append(
        f"  trials = {total} × {args.n_configs} configs × {args.seeds} seeds × "
        f"{args.temps} temps = {trials}"
    )
    empty_buckets = [lb for lb in labels if pooled[lb] == 0]
    if empty_buckets:
        lines.append(
            f"  NOTE: no coverage in length bucket(s): {', '.join(empty_buckets)} "
            f"(length sweep will have gaps there)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--groups", default=",".join(available_groups()), help="Comma-separated."
    )
    parser.add_argument("--max-per-group", type=int, default=None)
    parser.add_argument(
        "--voice-mode", default="clone", choices=["auto", "preset", "clone"]
    )
    parser.add_argument(
        "--n-configs",
        type=int,
        default=5,
        help="Configs incl. baseline (for the trial estimate).",
    )
    parser.add_argument("--seeds", type=int, default=1, help="Seed count.")
    parser.add_argument("--temps", type=int, default=1, help="Temperature count.")
    parser.add_argument(
        "--check-audio",
        action="store_true",
        help="Also verify ref/ground-truth audio files exist on disk.",
    )
    parser.add_argument("--out-md", default=None)
    args = parser.parse_args()

    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    per_group, all_warns = [], []
    for g in groups:
        try:
            items = iter_eval_items([g], args.max_per_group, args.data_dir)
        except Exception as exc:
            all_warns.append(f"{g}: FAILED to load — {exc}")
            per_group.append({"group": g, "n": 0})
            continue
        stats, warns = audit_group(g, items, args.voice_mode, args.check_audio)
        per_group.append(stats)
        all_warns.extend(warns)

    lines: list[str] = []
    render(per_group, args, lines)
    if all_warns:
        lines.append("\n## ⚠ WARNINGS\n")
        for w in all_warns:
            lines.append(f"  - {w}")
    else:
        lines.append(
            "\n## ✓ No warnings — all groups loaded and coverage looks complete."
        )

    report = "\n".join(lines)
    print(report)
    if args.out_md:
        os.makedirs(os.path.dirname(args.out_md) or ".", exist_ok=True)
        with open(args.out_md, "w", encoding="utf-8") as fh:
            fh.write(report + "\n")
        print(f"\nWrote {args.out_md}")


if __name__ == "__main__":
    main()
