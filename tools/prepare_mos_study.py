"""Prepare a MOS listening-test clip set (VALL-E KV-quant) for Qualtrics.

Selects clean librispeech_pc sentences and, for each, gathers the 6 systems
(natural ground-truth + fp16 + K4V4@0 + K4V2@0 + K3V3@0 + K2V2@0), loudness-
normalizes every clip, anonymizes filenames (so the system is not guessable),
and writes a manifest mapping each clip_id back to sentence/system plus the
objective CER/WER (for the later MOS-vs-CER correlation).

Runs on the box where the wavs live (grid pl0 output) + ground-truth audio.
  python tools/prepare_mos_study.py \
      --scores results/grid_pl0_scores.csv \
      --audio-dir models/VALL-E-X/benchmarks/outputs/grid_pl0 \
      --data-dir data --n-sentences 12 --out mos_study

Notes / caveats written into mos_study/README:
  * pl0 audio (deployment config); seed 0.
  * natural anchor = LibriSpeech test-clean recording (16 kHz), resampled to
    24 kHz for format parity — it is band-limited vs the 24 kHz synthetic, so
    it anchors "is it real speech" (attention check) more than a strict fidelity
    ceiling. Note this in the writeup.
  * clips are RMS-normalized to a common level (MOS is biased by loudness).
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402

from turboquant.eval_sentences import iter_eval_items  # noqa: E402

GROUP = "librispeech_pc"
SYSTEMS = ["natural", "fp16", "K4V4@0", "K4V2@0", "K3V3@0", "K2V2@0"]
TARGET_SR = 24000
TARGET_RMS_DBFS = -20.0
SEED = 0  # RNG seed for the (deterministic) clip-order shuffle


def _rms_normalize(wav: np.ndarray) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float64)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    rms = float(np.sqrt(np.mean(wav ** 2))) + 1e-9
    wav = wav * (10 ** (TARGET_RMS_DBFS / 20) / rms)
    peak = float(np.abs(wav).max())
    if peak > 0.99:
        wav = wav * (0.99 / peak)  # prevent clipping after gain
    return wav


def _gen_path(audio_dir: str, idx: int, config: str) -> str:
    return os.path.join(audio_dir,
                        f"vallex_{GROUP}_{idx}_sampling_s0_{config}.wav")


def _select_sentences(scores: str, n: int) -> list[int]:
    """Clean, medium-length sentence idxs: low fp16 CER, 4-9 s."""
    d = pd.read_csv(scores)
    fp = d[(d["group"] == GROUP) & (d["config"].astype(str).str.lower() == "fp16")]
    fp = fp[(fp["dur_s"] >= 4.0) & (fp["dur_s"] <= 9.0)]
    fp = fp.sort_values("cer")  # cleanest first
    idxs = fp["idx"].astype(int).tolist()[:n]
    if len(idxs) < n:
        raise SystemExit(f"only {len(idxs)} clean sentences found; lower --n-sentences")
    return sorted(idxs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", default="results/grid_pl0_scores.csv")
    ap.add_argument("--audio-dir",
                    default="models/VALL-E-X/benchmarks/outputs/grid_pl0")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--n-sentences", type=int, default=12)
    ap.add_argument("--out", default="mos_study")
    args = ap.parse_args()

    idxs = _select_sentences(args.scores, args.n_sentences)
    items = iter_eval_items([GROUP], None, args.data_dir)
    scores = pd.read_csv(args.scores)

    clips_dir = os.path.join(args.out, "clips")
    os.makedirs(clips_dir, exist_ok=True)

    def _cer_wer(idx, system):
        if system == "natural":
            return "", ""
        r = scores[(scores["group"] == GROUP) & (scores["idx"] == idx)
                   & (scores["config"] == system)]
        if r.empty:
            return "", ""
        return float(r["cer"].iloc[0]), float(r["wer"].iloc[0])

    # Build every (sentence, system) clip, then shuffle → anonymous ids.
    plan = [(idx, sys_) for idx in idxs for sys_ in SYSTEMS]
    rng = random.Random(SEED)
    rng.shuffle(plan)

    manifest = []
    missing = []
    for n, (idx, system) in enumerate(plan, start=1):
        clip_id = f"c{n:03d}"
        if system == "natural":
            gt = getattr(items[idx], "ground_truth_audio", None)
            if not gt or not os.path.exists(gt):
                missing.append((idx, system)); continue
            wav, _ = librosa.load(gt, sr=TARGET_SR, mono=True)
        else:
            src = _gen_path(args.audio_dir, idx, system)
            if not os.path.exists(src):
                missing.append((idx, system)); continue
            wav, sr = sf.read(src)
            if sr != TARGET_SR:
                wav = librosa.resample(np.asarray(wav, float), orig_sr=sr,
                                       target_sr=TARGET_SR)
        wav = _rms_normalize(wav)
        sf.write(os.path.join(clips_dir, f"{clip_id}.wav"), wav, TARGET_SR)
        cer, wer = _cer_wer(idx, system)
        manifest.append({
            "clip_id": clip_id, "sentence_idx": idx, "system": system,
            "text": items[idx].text, "orig_cer": cer, "orig_wer": wer,
        })

    with open(os.path.join(args.out, "manifest.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["clip_id", "sentence_idx", "system",
                                           "text", "orig_cer", "orig_wer"])
        w.writeheader()
        w.writerows(manifest)

    with open(os.path.join(args.out, "README.txt"), "w", encoding="utf-8") as fh:
        fh.write(
            "MOS listening study — VALL-E KV-cache quantization\n"
            f"{len(idxs)} sentences x {len(SYSTEMS)} systems = {len(manifest)} clips.\n"
            "pl0 audio, seed 0. RMS-normalized to a common level.\n"
            "clip_id is anonymized; manifest.csv maps clip_id -> sentence/system.\n"
            "natural = LibriSpeech test-clean (16 kHz) resampled to 24 kHz — "
            "band-limited vs synthetic; use as attention check / upper anchor.\n"
            "Attention checks: the 'natural' clips should score high; the "
            "'K2V2@0' clips should score low. Drop listeners failing both.\n"
        )

    print(f"wrote {len(manifest)} clips -> {clips_dir}")
    print(f"manifest -> {os.path.join(args.out, 'manifest.csv')}")
    if missing:
        print(f"WARNING: {len(missing)} clips missing (sentence,system): {missing[:8]}")


if __name__ == "__main__":
    main()
