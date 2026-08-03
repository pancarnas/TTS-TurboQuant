"""Feasibility probe for the WER-stratified MOS/SMOS sentence draw (read-only).

Mirrors prepare_mos_study.py's completeness + speaker logic so the reported
ceilings match what a build would actually select. For each WER band it prints the
number of candidate sentences, how many are *complete* (all 5 configs + ground
truth + reference present), and how many DISTINCT reference speakers those span;
plus the global unique-speaker ceiling and a breakdown of why sentences are
incomplete. Run on the box where the clone-grid wavs + LibriSpeech audio live.

  python tools/mos_feasibility.py \
      --scores results/clone/grid_clone_pl0_scores.csv \
      --audio-dir models/VALL-E-X/benchmarks/outputs/grid_clone_pl0 --data-dir data

The two decision numbers: TOTAL unique speakers among complete (hard ceiling on a
--unique-speakers design) and the incomplete-reason counts (if most are
'missing:...' the clone grid lacks configs, so completeness — not speakers — is the
bottleneck).
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from turboquant.eval_sentences import iter_eval_items  # noqa: E402

GROUP = "librispeech_pc"
# synthesized configs a sentence must have (natural = ground truth, checked apart)
SYSTEMS = ["fp16", "K4V4@0", "K4V2@0", "K3V3@0", "K2V2@0"]
DEFAULT_STRATA = "0:0.05,0.05:0.15,0.15:0.30,0.30:9"


def _gen_path(audio_dir: str, idx: int, config: str) -> str:
    return os.path.join(audio_dir,
                        f"vallex_{GROUP}_{idx}_sampling_s0_{config}.wav")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", default="results/clone/grid_clone_pl0_scores.csv")
    ap.add_argument("--audio-dir",
                    default="models/VALL-E-X/benchmarks/outputs/grid_clone_pl0")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--bands", default=DEFAULT_STRATA,
                    help="WER bands 'lo:hi,lo:hi,...' to report (no quotas)")
    args = ap.parse_args()

    items = iter_eval_items([GROUP], None, args.data_dir)

    def why_incomplete(i: int) -> str:
        if i >= len(items):
            return "no-item"
        gt = getattr(items[i], "ground_truth_audio", None)
        ref = getattr(items[i], "ref_audio", None)
        if not gt or not os.path.exists(gt):
            return "no-gt"
        if not ref or not os.path.exists(ref):
            return "no-ref"
        miss = [c for c in SYSTEMS if not os.path.exists(_gen_path(args.audio_dir, i, c))]
        return "ok" if not miss else "missing:" + ",".join(miss)

    def spk(i: int) -> str:
        ref = (getattr(items[i], "ref_audio", None)
               or getattr(items[i], "ground_truth_audio", None))
        return os.path.basename(ref).split("-")[0] if ref else str(i)

    d = pd.read_csv(args.scores)
    fp = d[(d["group"] == GROUP) & (d["config"].astype(str).str.lower() == "fp16")]
    fp = fp[(fp["dur_s"] >= 4.0) & (fp["dur_s"] <= 9.0)]

    bands = []
    for part in args.bands.split(","):
        lo, hi = part.split(":")
        bands.append((float(lo), float(hi)))

    print("=== per WER band (complete = all 5 configs + gt + ref present) ===")
    for lo, hi in bands:
        idxs = [int(i) for i in fp[(fp["wer"] >= lo) & (fp["wer"] < hi)]["idx"]]
        comp = [i for i in idxs if why_incomplete(i) == "ok"]
        print(f"  WER [{lo:g},{hi:g}): {len(idxs):3d} sent | {len(comp):3d} complete | "
              f"{len(set(map(spk, comp))):3d} unique spk among complete")

    allc = [int(i) for i in fp["idx"] if why_incomplete(int(i)) == "ok"]
    print(f"TOTAL complete 4-9s: {len(allc)} | unique speakers: "
          f"{len(set(map(spk, allc)))}")
    reasons = Counter(why_incomplete(int(i)) for i in fp["idx"]
                      if why_incomplete(int(i)) != "ok")
    print(f"incomplete reasons (all 4-9s): {dict(reasons)}")


if __name__ == "__main__":
    main()
