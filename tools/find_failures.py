"""List sentences where a config (default: baseline) fails — triage bad refs / text.

A high BASELINE (uncompressed) CER means the clone already failed for that
sentence — usually a noisy/short reference prompt or un-synthesizable text — so
that whole cell is a shaky basis for the compression comparison. This maps each
trial row back to its sentence text + reference clip (via the same
``iter_eval_items`` order the benchmark used) and prints the failures, sorted
worst-first, so you can decide which to exclude or re-check.

Run (box, after/while the run produces CSVs):
  python tools/find_failures.py --trials-glob 'results/qwen_trials_shard*_clone_rw0_*.csv' \
      --data-dir data --groups seedtts_en,librispeech_pc,libritts_long,ellav_hard \
      --max-per-group 100 --threshold 0.10
"""

from __future__ import annotations

import argparse
import glob
import os

import pandas as pd

from turboquant.eval_sentences import iter_eval_items


def _index_items(groups, max_per_group, data_dir):
    """(group, idx_within_group) -> EvalItem, matching the benchmark's order."""
    table = {}
    for g in groups:
        try:
            items = iter_eval_items([g], max_per_group, data_dir)
        except Exception as exc:
            print(f"  (could not load group {g}: {exc})")
            continue
        for i, it in enumerate(items):
            table[(g, i)] = it
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials-glob", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--groups", required=True, help="Comma-separated (run's groups)."
    )
    parser.add_argument("--max-per-group", type=int, default=None)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.10,
        help="Flag CER above this (default 0.10).",
    )
    parser.add_argument(
        "--config",
        default="baseline",
        help="Substring of the config to inspect (default 'baseline').",
    )
    parser.add_argument("--out-csv", default=None)
    args = parser.parse_args()

    paths = sorted(glob.glob(args.trials_glob))
    if not paths:
        raise SystemExit(f"no CSVs match {args.trials_glob}")
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df = df[df["config"].astype(str).str.contains(args.config, case=False)]
    df = df.dropna(subset=["cer"])

    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    items = _index_items(groups, args.max_per_group, args.data_dir)

    rows = []
    for _, r in df.iterrows():
        it = items.get((r["group"], int(r["idx"])))
        ref = getattr(it, "ref_audio", None) if it else None
        rows.append(
            {
                "group": r["group"],
                "idx": int(r["idx"]),
                "cer": float(r["cer"]),
                "ref_audio": os.path.basename(ref) if ref else "",
                "text": (it.text[:70] if it else "?"),
            }
        )
    out = pd.DataFrame(rows).sort_values("cer", ascending=False)
    fails = out[out["cer"] > args.threshold]

    n = len(out)
    print(
        f"\n{args.config} CER over {n} sentences: "
        f"mean={out['cer'].mean():.3f}  median={out['cer'].median():.3f}  "
        f"failures(>{args.threshold:.0%})={len(fails)}/{n}\n"
    )
    print(f"{'group':<14}{'idx':>4}{'CER':>8}  {'ref_clip':<28} text")
    print("-" * 100)
    for _, r in fails.iterrows():
        print(
            f"{r['group']:<14}{r['idx']:>4}{r['cer']:>8.3f}  "
            f"{r['ref_audio']:<28} {r['text']}"
        )

    if args.out_csv:
        out.to_csv(args.out_csv, index=False)
        print(f"\nWrote full per-sentence {args.config} CER table → {args.out_csv}")
    print(
        "\nHigh baseline CER ⇒ likely a noisy/short reference clip OR un-synthesizable "
        "text. To confirm a dirty REFERENCE specifically, run tools/validate_eval_set.py "
        "(it ASRs the prompt clip itself + estimates SNR)."
    )


if __name__ == "__main__":
    main()
