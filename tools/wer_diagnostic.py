"""Diagnose high WER: show reference vs ASR transcript (normalized), per sentence.

The scorer already saves the Whisper transcript per row, so we can inspect what
jiwer actually compared WITHOUT re-running or listening first. For each sentence
it prints the normalized reference and normalized hypothesis (exactly the two
strings jiwer sees) plus WER/CER, and recomputes WER from the normalized pair to
confirm the stored value. This distinguishes:
  * real model failure (hallucination/skips) -> HYP words wildly differ from REF
  * a metric/normalization artifact          -> HYP ~ REF yet WER high

  python tools/wer_diagnostic.py --scores results/seed_pl0_scores.csv \
      --config fp16 --group librispeech_pc --data-dir data --n 12
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from turboquant.eval_sentences import iter_eval_items  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--config", default="fp16", help="which config's transcripts")
    ap.add_argument("--group", default="librispeech_pc")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--n", type=int, default=12, help="worst-N sentences to print")
    args = ap.parse_args()

    from jiwer import wer as jiwer_wer
    from whisper.normalizers import EnglishTextNormalizer
    norm = EnglishTextNormalizer()

    d = pd.read_csv(args.scores)
    d = d[(d["group"] == args.group) & (d["config"] == args.config)]
    if d.empty:
        raise SystemExit(f"no rows for group={args.group} config={args.config}")
    # one row per sentence (first seed)
    d = d.sort_values("seed").drop_duplicates("idx")

    items = iter_eval_items([args.group], None, args.data_dir)

    recs = []
    for _, r in d.iterrows():
        idx = int(r["idx"])
        ref_raw = items[idx].text
        hyp_raw = "" if pd.isna(r.get("transcript")) else str(r["transcript"])
        rn, hn = norm(ref_raw), norm(hyp_raw)
        recomputed = float(jiwer_wer(rn, hn)) if rn else 0.0
        recs.append({
            "idx": idx, "wer": float(r["wer"]), "cer": float(r["cer"]),
            "recomputed_wer": recomputed,
            "ref_norm": rn, "hyp_norm": hn,
            "ref_words": len(rn.split()), "hyp_words": len(hn.split()),
        })
    df = pd.DataFrame(recs)

    print(f"== {args.group} / {args.config}  ({len(df)} sentences) ==")
    print(f"WER: mean {df.wer.mean():.3f}  median {df.wer.median():.3f}  "
          f">0.5: {(df.wer > 0.5).mean():.1%}")
    # stored vs recomputed WER should match (catches a scoring bug)
    mism = (df["wer"] - df["recomputed_wer"]).abs()
    print(f"stored-vs-recomputed WER max diff: {mism.max():.4f} "
          f"({'OK' if mism.max() < 1e-3 else 'MISMATCH — scoring bug!'})")
    # length ratio: hyp much shorter/longer than ref => skips/hallucination
    df["len_ratio"] = df["hyp_words"] / df["ref_words"].clip(lower=1)
    print(f"hyp/ref word-count ratio: mean {df.len_ratio.mean():.2f} "
          f"(≈1 = same length; <1 = skipping; >1 = inserting)")

    print(f"\n-- worst {args.n} sentences (highest WER) --")
    for _, r in df.sort_values("wer", ascending=False).head(args.n).iterrows():
        print(f"\nidx {int(r.idx)}  WER {r.wer:.2f}  CER {r.cer:.2f}  "
              f"({int(r.ref_words)}->{int(r.hyp_words)} words)")
        print(f"  REF: {r.ref_norm}")
        print(f"  HYP: {r.hyp_norm}")


if __name__ == "__main__":
    main()
