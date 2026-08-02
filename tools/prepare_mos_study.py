"""Prepare a MOS + SMOS listening-test clip set (VALL-E KV-quant) for Qualtrics.

Selects clean librispeech_pc sentences and, for each, gathers the 6 systems
(natural ground-truth + fp16 + K4V4@0 + K4V2@0 + K3V3@0 + K2V2@0). With
--with-similarity it ALSO emits each sentence's reference / cloning-prompt clip,
so the survey can pair reference + stimulus for a speaker-similarity (SMOS)
rating. Every clip is loudness-normalized, resampled to 24 kHz, and given an
anonymized id so the system is not guessable; the manifest maps each id back to
sentence / system / block plus the objective CER/WER.

Runs on the box where the wavs live (clone grid pl0 output) + the LibriSpeech
ground-truth/reference audio.

  python tools/prepare_mos_study.py --with-similarity --blocks 2 \
      --n-sentences 20 --scores results/grid_cl_pl0_scores.csv \
      --audio-dir models/VALL-E-X/benchmarks/outputs/grid_cl_pl0 \
      --data-dir data --out mos_study_clone

A sentence is only selected if all 6 clips exist (ground truth + 5 systems), so
the design stays balanced; --with-similarity additionally needs the item's
reference recording. After auditioning, rerun with --exclude-idx N [N ...] to
swap out any sentence whose fp16/natural/reference clip has a defect unrelated to
quantization; the next-cleanest candidates fill in.

Notes written into the README:
  * pl0 audio (deployment config); seed 0.
  * natural anchor = LibriSpeech recording (resampled to 24 kHz), band-limited vs
    the synthetic clips, so it anchors "is this real speech" (attention check)
    more than a strict fidelity ceiling.
  * the reference clip is the cloning prompt (a DIFFERENT utterance by the same
    speaker), used only for the similarity rating.
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


def _load24k(path: str, max_sec: float | None = None) -> np.ndarray:
    """Load any wav/flac as mono 24 kHz float (librosa resamples as needed)."""
    wav, _ = librosa.load(path, sr=TARGET_SR, mono=True, duration=max_sec)
    return wav


def _trim_onset(wav: np.ndarray, sr: int) -> np.ndarray:
    """Remove the VALL-E-X onset burst + leading silence from a synthesized clip.

    VALL-E-X emits a brief high-energy burst at the very start followed by a
    stretch of silence before the speech. This detects the speech onset (first
    sustained-energy region preceded by quiet, so the burst is not mistaken for
    speech), trims to ~200 ms before it (always at least 150 ms so any burst is
    removed; speech never starts that early), and applies a 20 ms raised-cosine
    fade-in. Natural/reference clips are real recordings and must NOT be trimmed.
    """
    w = np.asarray(wav, dtype=np.float64)
    win = int(0.010 * sr)
    nf = len(w) // win
    if nf < 6:
        return wav
    e = np.array([np.sqrt(np.mean(w[i * win:(i + 1) * win] ** 2)) for i in range(nf)])
    thr = 0.10 * (e[20:].max() if nf > 20 else e.max())
    onset = -1
    for i in range(2, nf):
        if e[i] > thr and np.mean(e[i:i + 8] > thr) >= 0.6:
            pre = e[max(0, i - 14):max(1, i - 2)]
            if len(pre) == 0 or np.mean(pre < thr) >= 0.7:
                onset = i * 10  # ms
                break
    trim_ms = max(150, onset - 200) if onset > 0 else 150
    cut = min(int(trim_ms / 1000 * sr), max(0, len(wav) - sr))  # keep >=1 s
    nw = np.asarray(wav[cut:], dtype=np.float64)
    nfade = int(0.020 * sr)
    if len(nw) > nfade:
        nw[:nfade] *= 0.5 * (1 - np.cos(np.pi * np.arange(nfade) / nfade))
    return nw


def _select_sentences(scores: str, n: int, exclude: set[int],
                      complete) -> list[int]:
    """Clean, medium-length sentence idxs: low fp16 CER, 4-9 s, cleanest first.

    Only accepts a sentence when every required clip exists, so the design stays
    balanced. Returns idxs ordered by fp16 CER (cleanest first) so the caller can
    assign balanced blocks.
    """
    d = pd.read_csv(scores)
    fp = d[(d["group"] == GROUP) & (d["config"].astype(str).str.lower() == "fp16")]
    fp = fp[(fp["dur_s"] >= 4.0) & (fp["dur_s"] <= 9.0)]
    fp = fp.sort_values("cer")  # cleanest first
    idxs, skipped = [], []
    for idx in fp["idx"].astype(int):
        if idx in exclude:
            continue
        if not complete(idx):
            skipped.append(idx)
            continue
        idxs.append(idx)
        if len(idxs) == n:
            break
    if skipped:
        print(f"skipped {len(skipped)} candidates with missing clips: {skipped}")
    if len(idxs) < n:
        raise SystemExit(f"only {len(idxs)} usable sentences found; lower --n-sentences")
    return idxs  # cleanest-first order


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", default="results/grid_cl_pl0_scores.csv")
    ap.add_argument("--audio-dir",
                    default="models/VALL-E-X/benchmarks/outputs/grid_cl_pl0")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--n-sentences", type=int, default=20)
    ap.add_argument("--with-similarity", action="store_true",
                    help="also emit each sentence's reference/cloning-prompt clip "
                         "so the survey can pair reference + stimulus (SMOS)")
    ap.add_argument("--blocks", type=int, default=1,
                    help="split the sentences into N balanced blocks (recorded in "
                         "the manifest) to cap per-listener load")
    ap.add_argument("--ref-max-sec", type=float, default=10.0,
                    help="cap the reference clip length (seconds)")
    ap.add_argument("--exclude-idx", type=int, nargs="*", default=[],
                    help="sentence idxs to skip (e.g. rejected during audition)")
    ap.add_argument("--out", default="mos_study")
    args = ap.parse_args()

    items = iter_eval_items([GROUP], None, args.data_dir)

    def _complete(idx: int) -> bool:
        if idx >= len(items):
            return False
        gt = getattr(items[idx], "ground_truth_audio", None)
        if not gt or not os.path.exists(gt):
            return False
        if args.with_similarity:
            ref = getattr(items[idx], "ref_audio", None)
            if not ref or not os.path.exists(ref):
                return False
        return all(os.path.exists(_gen_path(args.audio_dir, idx, s))
                   for s in SYSTEMS if s != "natural")

    idxs = _select_sentences(args.scores, args.n_sentences,
                             set(args.exclude_idx), _complete)
    scores = pd.read_csv(args.scores)

    # balanced blocks: round-robin over the cleanest-first order
    block_of = {idx: (rank % max(args.blocks, 1)) for rank, idx in enumerate(idxs)}

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

    def _write(clip_id, wav):
        sf.write(os.path.join(clips_dir, f"{clip_id}.wav"),
                 _rms_normalize(wav), TARGET_SR)

    manifest, missing = [], []

    # reference clips (one per sentence) — assigned r-ids, used for SMOS
    ref_id = {}
    if args.with_similarity:
        for n, idx in enumerate(idxs, start=1):
            rid = f"r{n:03d}"
            ref = getattr(items[idx], "ref_audio", None)
            if not ref or not os.path.exists(ref):
                missing.append((idx, "reference")); continue
            _write(rid, _load24k(ref, max_sec=args.ref_max_sec))
            ref_id[idx] = rid
            manifest.append({
                "clip_id": rid, "block": block_of[idx], "role": "reference",
                "system": "reference", "sentence_idx": idx, "ref_clip_id": "",
                "text": items[idx].text, "orig_cer": "", "orig_wer": "",
            })

    # stimuli — shuffled within each block, then anonymized c001..cNNN
    by_block: dict[int, list[tuple[int, str]]] = {}
    for idx in idxs:
        for s in SYSTEMS:
            by_block.setdefault(block_of[idx], []).append((idx, s))
    rng = random.Random(SEED)
    cnt = 0
    for b in sorted(by_block):
        lst = by_block[b][:]
        rng.shuffle(lst)
        for idx, system in lst:
            cnt += 1
            cid = f"c{cnt:03d}"
            src = (getattr(items[idx], "ground_truth_audio", None)
                   if system == "natural" else _gen_path(args.audio_dir, idx, system))
            if not src or not os.path.exists(src):
                missing.append((idx, system)); continue
            wav = _load24k(src)
            if system != "natural":          # trim VALL-E onset artifact on synth only
                wav = _trim_onset(wav, TARGET_SR)
            _write(cid, wav)
            cer, wer = _cer_wer(idx, system)
            manifest.append({
                "clip_id": cid, "block": b, "role": "stimulus", "system": system,
                "sentence_idx": idx, "ref_clip_id": ref_id.get(idx, ""),
                "text": items[idx].text, "orig_cer": cer, "orig_wer": wer,
            })

    cols = ["clip_id", "block", "role", "system", "sentence_idx", "ref_clip_id",
            "text", "orig_cer", "orig_wer"]
    with open(os.path.join(args.out, "manifest.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(manifest)

    n_stim = sum(1 for m in manifest if m["role"] == "stimulus")
    n_ref = sum(1 for m in manifest if m["role"] == "reference")
    with open(os.path.join(args.out, "README.txt"), "w", encoding="utf-8") as fh:
        fh.write(
            "MOS + SMOS listening study - VALL-E KV-cache quantization\n"
            f"{len(idxs)} sentences x {len(SYSTEMS)} systems = {n_stim} stimuli"
            + (f" + {n_ref} reference clips" if args.with_similarity else "") + ".\n"
            f"{args.blocks} block(s); pl0 audio, seed 0; RMS-normalized to a common level.\n"
            "clip_id is anonymized; manifest.csv maps clip_id -> block/role/system/"
            "sentence, and each stimulus's ref_clip_id points at its sentence's "
            "reference clip for the similarity rating.\n"
            "Naturalness (MOS): 5-point ACR, 'rate the naturalness/quality'.\n"
            "Similarity (SMOS): play reference (r***) then stimulus, 'how similar "
            "is the speaker'.\n"
            "natural = LibriSpeech recording resampled to 24 kHz (band-limited vs "
            "synthetic) - attention check / upper anchor.\n"
            "Attention checks: 'natural' clips score high; 'K2V2@0' clips score low. "
            "Drop listeners failing both.\n"
        )

    print(f"wrote {n_stim} stimuli" + (f" + {n_ref} references" if args.with_similarity else "")
          + f" -> {clips_dir}")
    print(f"blocks: " + ", ".join(f"{b}={sum(1 for i in idxs if block_of[i]==b)} sentences"
                                  for b in sorted(set(block_of.values()))))
    print(f"manifest -> {os.path.join(args.out, 'manifest.csv')}")
    if missing:
        print(f"WARNING: {len(missing)} clips missing (sentence,system): {missing[:8]}")


if __name__ == "__main__":
    main()
