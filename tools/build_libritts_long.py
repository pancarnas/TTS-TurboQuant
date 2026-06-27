"""Build the long-context set by concatenating LibriTTS-R utterances.

Consumes the fetched LibriTTS-R test-clean (``<data-dir>/libritts_r``), groups
utterances by (speaker, chapter), and uses the pure, tested
``plan_long_concatenations`` to form graduated-length passages (256/512/1024/
2048 talker tokens). For each passage it stitches the member wavs into one real
recording (the ground truth) and writes a seed-tts-style manifest the loader
reads:

  ``<data-dir>/libritts_long/manifest.lst``  — ``id|ref_text|ref_wav|target_text|gt_wav``
  ``<data-dir>/libritts_long/gt/*.wav``       — concatenated ground-truth audio

``ref_wav`` is a same-speaker utterance NOT in the passage (chosen by the planner)
so the clone prompt never leaks the target. Run AFTER
``fetch_eval_data.py --fetch-libritts``. Box-only (needs the audio + soundfile).

Run: ``python tools/build_libritts_long.py --data-dir data``
"""

from __future__ import annotations

import argparse
import glob
import os

from turboquant.eval_sentences import LONG_BUCKETS, plan_long_concatenations


def _read_text(wav_path: str) -> str:
    """LibriTTS ships ``<utt>.normalized.txt`` (preferred) / ``.original.txt``."""
    for suffix in (".normalized.txt", ".original.txt"):
        cand = wav_path.rsplit(".wav", 1)[0] + suffix
        if os.path.exists(cand):
            with open(cand, encoding="utf-8") as fh:
                return fh.read().strip()
    return ""


def collect_utterances(libritts_root: str) -> list[dict]:
    """Scan LibriTTS-R wavs → records {speaker, chapter, idx, text, wav}.

    Filenames are ``<spk>_<chapter>_<utt>_<seg>.wav``; idx orders utterances within
    a chapter. Utterances with no transcript file are skipped.
    """
    records: list[dict] = []
    for wav in glob.glob(os.path.join(libritts_root, "**", "*.wav"), recursive=True):
        stem = os.path.basename(wav)[: -len(".wav")]
        parts = stem.split("_")
        if len(parts) < 4:
            continue
        spk, chapter, utt, seg = parts[0], parts[1], parts[2], parts[3]
        text = _read_text(wav)
        if not text:
            continue
        try:
            idx = int(utt) * 1000 + int(seg)
        except ValueError:
            idx = 0
        records.append(
            {"speaker": spk, "chapter": chapter, "idx": idx, "text": text, "wav": wav}
        )
    return records


def _concat_wavs(member_wavs: list[str], out_path: str) -> None:
    import numpy as np
    import soundfile as sf

    chunks, sr = [], None
    for w in member_wavs:
        audio, this_sr = sf.read(w)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        sr = sr or this_sr
        chunks.append(audio)
    if chunks:
        sf.write(out_path, np.concatenate(chunks), sr or 24000)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--max-per-bucket",
        type=int,
        default=None,
        help="Cap passages per length bucket (keeps the run tractable).",
    )
    args = parser.parse_args()

    libritts_root = os.path.join(args.data_dir, "libritts_r")
    out_dir = os.path.join(args.data_dir, "libritts_long")
    gt_dir = os.path.join(out_dir, "gt")
    os.makedirs(gt_dir, exist_ok=True)

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    utts = collect_utterances(libritts_root)
    print(f"collected {len(utts)} LibriTTS-R utterances")
    plan = plan_long_concatenations(utts, buckets=LONG_BUCKETS)

    manifest = os.path.join(out_dir, "manifest.lst")
    per_bucket: dict[int, int] = {}
    written = 0
    plan_iter = tqdm(plan, desc="concat", unit="passage") if tqdm is not None else plan
    with open(manifest, "w", encoding="utf-8") as fh:
        for rec in plan_iter:
            b = rec["bucket"]
            if args.max_per_bucket and per_bucket.get(b, 0) >= args.max_per_bucket:
                continue
            if not rec["ref_wav"]:
                continue  # no same-speaker prompt available — skip
            n = per_bucket.get(b, 0)
            per_bucket[b] = n + 1
            gt_path = os.path.join(gt_dir, f"long_{b}_{n}.wav")
            _concat_wavs(rec["member_wavs"], gt_path)
            # seed-tts-style row: id|ref_text|ref_wav|target_text|gt_wav (abs paths).
            fh.write(
                f"long_{b}_{n}|{rec['ref_text']}|{rec['ref_wav']}|"
                f"{rec['target_text']}|{gt_path}\n"
            )
            written += 1

    print(f"wrote {written} long passages → {manifest}")
    for b in sorted(per_bucket):
        print(f"  bucket {b}: {per_bucket[b]} passages")


if __name__ == "__main__":
    main()
