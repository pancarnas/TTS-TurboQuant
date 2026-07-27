"""Collect the worst-WER sentences into one folder for local listening.

For the N highest-WER sentences of a config, copies the generated wav (+ the
natural ground-truth recording, + optionally other configs of the same sentence
for A/B) into --out, with readable names, and writes listen.txt showing the
reference text and the ASR transcript per sentence. scp the folder and listen
while reading REF vs HYP.

  python tools/collect_listen.py --scores results/seed_pl0_scores.csv \
      --config fp16 --group librispeech_pc --data-dir data \
      --audio-dir models/VALL-E-X/benchmarks/outputs/seed_pl0 \
      --n 12 --also-configs K4V2@0,K2V2@0 --out listen_wers
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from turboquant.eval_sentences import iter_eval_items  # noqa: E402


def _wavname(group, idx, config):
    return f"vallex_{group}_{idx}_sampling_s0_{config}.wav"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--config", default="fp16")
    ap.add_argument("--group", default="librispeech_pc")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--audio-dir",
                    default="models/VALL-E-X/benchmarks/outputs/seed_pl0")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--also-configs", default="",
                    help="comma configs of the SAME sentences to copy for A/B "
                    "(e.g. K4V2@0,K2V2@0)")
    ap.add_argument("--out", default="listen_wers")
    args = ap.parse_args()

    d = pd.read_csv(args.scores)
    d = d[(d["group"] == args.group) & (d["config"] == args.config)]
    if d.empty:
        raise SystemExit(f"no rows for {args.group}/{args.config}")
    d = d.sort_values("seed").drop_duplicates("idx")
    worst = d.sort_values("wer", ascending=False).head(args.n)

    items = iter_eval_items([args.group], None, args.data_dir)
    also = [c.strip() for c in args.also_configs.split(",") if c.strip()]
    os.makedirs(args.out, exist_ok=True)

    lines, missing = [], []
    for _, r in worst.iterrows():
        idx = int(r["idx"])
        tag = f"idx{idx}_wer{int(round(float(r['wer'])*100)):03d}"
        # generated wav for the target config
        src = os.path.join(args.audio_dir, _wavname(args.group, idx, args.config))
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out, f"{tag}_{args.config}.wav"))
        else:
            missing.append(src)
        # natural ground-truth
        gt = getattr(items[idx], "ground_truth_audio", None)
        if gt and os.path.exists(gt):
            ext = os.path.splitext(gt)[1] or ".wav"
            shutil.copy(gt, os.path.join(args.out, f"{tag}_NATURAL{ext}"))
        # extra configs for A/B
        for c in also:
            s = os.path.join(args.audio_dir, _wavname(args.group, idx, c))
            if os.path.exists(s):
                shutil.copy(s, os.path.join(args.out, f"{tag}_{c}.wav"))
        lines.append(
            f"[{tag}]  WER {float(r['wer']):.2f}  CER {float(r['cer']):.2f}\n"
            f"  REF: {items[idx].text}\n"
            f"  HYP: {r.get('transcript', '')}\n"
        )

    with open(os.path.join(args.out, "listen.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"Worst {len(worst)} WER sentences — {args.group}/{args.config}\n\n")
        fh.write("\n".join(lines))

    print(f"copied clips -> {args.out}/  (+ listen.txt with REF/HYP)")
    if missing:
        print(f"WARNING: {len(missing)} generated wavs not found, e.g. {missing[:2]}")
    print("\nPull to your desktop with:")
    print(f"  scp -r s2801778@eddie.ecdf.ed.ac.uk:"
          f"$(pwd)/{args.out} ~/Desktop/")


if __name__ == "__main__":
    main()
