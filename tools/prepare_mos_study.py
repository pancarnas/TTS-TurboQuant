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

  python tools/prepare_mos_study.py --with-similarity --unique-speakers \
      --stratify-by wer --strata "0:0.05:15,0.05:0.15:10,0.15:0.30:5" --blocks 2 \
      --scores results/grid_cl_pl0_scores.csv \
      --audio-dir models/VALL-E-X/benchmarks/outputs/grid_cl_pl0 \
      --data-dir data --out mos_study_clone

Sentences are WER-stratified (fp16 word-error bands) so the set spans the model's
easy/medium/hard range rather than only its cleanest outputs, giving the analysis a
real error-rate spread; --unique-speakers keeps every reference speaker distinct.
Omit --strata to fall back to cleanest-first on --stratify-by, up to --n-sentences.

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
from collections import Counter

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


DEFAULT_STRATA = [(0.0, 0.05, 15), (0.05, 0.15, 10), (0.15, 0.30, 5)]


def _parse_strata(spec: str) -> list[tuple[float, float, int]]:
    """'lo:hi:quota,lo:hi:quota,...' -> [(lo, hi, quota), ...] on the fp16 metric."""
    bands = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        lo, hi, quota = part.split(":")
        bands.append((float(lo), float(hi), int(quota)))
    return bands


def _select_sentences(scores: str, n: int, exclude: set[int], complete,
                      speaker_of=None, unique_speakers: bool = False,
                      max_per_speaker: int | None = None,
                      stratify_by: str = "wer",
                      strata: list[tuple[float, float, int]] | None = None):
    """Choose 4-9 s sentence idxs, WER-stratified when `strata` is given.

    fp16 rows are filtered to 4-9 s. With `strata` (bands (lo, hi, quota) on the
    `stratify_by` metric), each band is filled with its lowest-metric *complete*
    sentences. A per-speaker cap can be applied (shared across bands): explicit
    `max_per_speaker`, else 1 when `unique_speakers`, else no cap. Without `strata`
    this falls back to cleanest-first on `stratify_by` (cer/wer), up to `n`.

    Returns (idxs, band_of) where band_of maps idx -> "lo-hi" band label ("" in the
    cleanest-first fallback) so the caller can record it in the manifest.
    """
    d = pd.read_csv(scores)
    fp = d[(d["group"] == GROUP) & (d["config"].astype(str).str.lower() == "fp16")]
    fp = fp[(fp["dur_s"] >= 4.0) & (fp["dur_s"] <= 9.0)]
    metric = stratify_by if stratify_by in ("wer", "cer") else "cer"

    # per-speaker cap: explicit --max-per-speaker, else 1 for --unique-speakers, else off
    cap = max_per_speaker if max_per_speaker is not None else (1 if unique_speakers else None)
    spk_count: Counter = Counter()

    def _take(cands: list[int], quota: int) -> tuple[list[int], list[int]]:
        """Up to `quota` complete idxs (respecting the per-speaker cap), in order."""
        picked, skipped_missing = [], []
        for idx in cands:
            if idx in exclude:
                continue
            if not complete(idx):
                skipped_missing.append(idx)
                continue
            if cap is not None and speaker_of is not None:
                spk = speaker_of(idx)
                if spk_count[spk] >= cap:
                    continue
                spk_count[spk] += 1
            picked.append(idx)
            if len(picked) == quota:
                break
        return picked, skipped_missing

    if strata:
        idxs: list[int] = []
        band_of: dict[int, str] = {}
        all_missing: list[int] = []
        for lo, hi, quota in strata:
            band = fp[(fp[metric] >= lo) & (fp[metric] < hi)].sort_values(metric)
            picked, missing = _take([int(x) for x in band["idx"]], quota)
            all_missing += missing
            label = f"{lo:g}-{hi:g}"
            for idx in picked:
                band_of[idx] = label
            idxs += picked
            tag = "" if len(picked) == quota else "  <-- SHORT"
            print(f"{metric} band [{lo:g},{hi:g}): {len(picked)}/{quota}{tag}")
        if all_missing:
            print(f"skipped {len(all_missing)} candidates with missing clips: "
                  f"{all_missing[:12]}")
        if len(idxs) < n:
            print(f"WARNING: {len(idxs)}/{n} sentences selected (bands underfilled); "
                  "adjust --strata or check feasibility")
        return idxs, band_of

    # cleanest-first fallback (original behaviour), sorted on the chosen metric.
    # n <= 0 means "take ALL complete sentences" (used when the audio dir was already
    # built to the exact target set, e.g. the speaker-diverse MOS campaign).
    order = fp.sort_values(metric)
    cand = [int(x) for x in order["idx"]]
    take_all = n <= 0
    picked, skipped = _take(cand, len(cand) if take_all else n)
    if skipped:
        print(f"skipped {len(skipped)} candidates with missing clips: {skipped}")
    if take_all:
        print(f"selected all {len(picked)} complete sentences")
    elif len(picked) < n:
        raise SystemExit(f"only {len(picked)} usable sentences found; lower --n-sentences")
    return picked, {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", default="results/grid_cl_pl0_scores.csv")
    ap.add_argument("--audio-dir",
                    default="models/VALL-E-X/benchmarks/outputs/grid_cl_pl0")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--n-sentences", type=int, default=20,
                    help="sentence count for cleanest-first selection; ignored when "
                         "--strata is given (N = sum of band quotas)")
    ap.add_argument("--stratify-by", choices=["wer", "cer", "none"], default="wer",
                    help="fp16 metric to rank/stratify sentences by (default wer)")
    ap.add_argument("--strata", default="",
                    help="WER/CER bands 'lo:hi:quota,...' e.g. "
                         "'0:0.05:15,0.05:0.15:10,0.15:0.30:5'; enables stratified "
                         "selection and sets N = sum of quotas")
    ap.add_argument("--unique-speakers", action="store_true",
                    help="never reuse a reference speaker (shorthand for "
                         "--max-per-speaker 1)")
    ap.add_argument("--max-per-speaker", type=int, default=None,
                    help="cap sentences per reference speaker (e.g. 3); overrides "
                         "--unique-speakers when both are given")
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

    def _speaker(idx: int) -> str:
        if idx >= len(items):
            return str(idx)
        ref = (getattr(items[idx], "ref_audio", None)
               or getattr(items[idx], "ground_truth_audio", None))
        return os.path.basename(ref).split("-")[0] if ref else str(idx)

    strata = _parse_strata(args.strata) if args.strata else None
    n_sent = sum(q for _, _, q in strata) if strata else args.n_sentences
    idxs, band_of = _select_sentences(
        args.scores, n_sent, set(args.exclude_idx), _complete,
        speaker_of=_speaker, unique_speakers=args.unique_speakers,
        max_per_speaker=args.max_per_speaker,
        stratify_by=args.stratify_by, strata=strata)
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
                "wer_band": band_of.get(idx, ""),
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
            _write(cid, _load24k(src))
            cer, wer = _cer_wer(idx, system)
            manifest.append({
                "clip_id": cid, "block": b, "role": "stimulus", "system": system,
                "sentence_idx": idx, "ref_clip_id": ref_id.get(idx, ""),
                "text": items[idx].text, "orig_cer": cer, "orig_wer": wer,
                "wer_band": band_of.get(idx, ""),
            })

    cols = ["clip_id", "block", "role", "system", "sentence_idx", "ref_clip_id",
            "text", "orig_cer", "orig_wer", "wer_band"]
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
