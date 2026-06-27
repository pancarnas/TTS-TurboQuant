"""Build the long-context set by concatenating LibriTTS-R utterances.

Consumes the fetched LibriTTS-R test-clean (``<data-dir>/libritts_r``), which the
HF dataset ships as **parquet shards** (audio + text are columns, not a flac
tree). Groups utterances by (speaker, chapter) and uses the pure, tested
``plan_long_concatenations`` to form graduated-length passages (256/512/1024/
2048 talker tokens). For each passage it stitches the member clips into one real
recording (the ground truth), writes the prompt clip, and emits a seed-tts-style
manifest the loader reads:

  ``<data-dir>/libritts_long/manifest.lst``  — ``id|ref_text|ref_wav|target_text|gt_wav``
  ``<data-dir>/libritts_long/gt/*.wav``       — concatenated ground-truth audio
  ``<data-dir>/libritts_long/refs/*.wav``     — same-speaker prompt clips

``ref_wav`` is a same-speaker utterance NOT in the passage (chosen by the
planner) so the clone prompt never leaks the target. Run AFTER
``fetch_eval_data.py --fetch-libritts``. Needs pandas+pyarrow + soundfile.

Run: ``python tools/build_libritts_long.py --data-dir data``
"""

from __future__ import annotations

import argparse
import glob
import io
import os

from turboquant.eval_sentences import LONG_BUCKETS, plan_long_concatenations


def _find_col(columns, *candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


def _uid_from(row, id_col) -> str:
    """Utterance id, e.g. ``1089_134686_000001_000001`` (strip dir/ext if a path)."""
    raw = str(row[id_col])
    base = os.path.basename(raw)
    return base[:-4] if base.endswith(".wav") else base


def _idx_from_uid(uid: str) -> int:
    parts = uid.split("_")
    try:
        return int(parts[2]) * 1000 + int(parts[3])
    except (IndexError, ValueError):
        return 0


def load_libritts_rows(libritts_root: str):
    """Read parquet shards → (records, audio map). ``records`` are planner inputs
    {speaker, chapter, idx, text, wav=uid}; ``audio`` maps uid → raw audio cell."""
    import pandas as pd

    files = sorted(
        glob.glob(os.path.join(libritts_root, "**", "*.parquet"), recursive=True)
    )
    if not files:
        raise SystemExit(
            f"no .parquet under {libritts_root} — run "
            "`make fetch-eval-data FETCH_LIBRITTS=1` first"
        )
    records: list[dict] = []
    audio: dict[str, object] = {}
    for f in files:
        df = pd.read_parquet(f)
        spk_c = _find_col(df.columns, "speaker_id", "speaker")
        chap_c = _find_col(df.columns, "chapter_id", "chapter")
        text_c = _find_col(
            df.columns, "text_normalized", "normalized_text", "text_original", "text"
        )
        id_c = _find_col(df.columns, "id", "utterance_id", "path", "file")
        audio_c = _find_col(df.columns, "audio", "wav")
        if not (text_c and audio_c and id_c):
            raise SystemExit(f"unexpected LibriTTS-R columns: {list(df.columns)}")
        for _, row in df.iterrows():
            uid = _uid_from(row, id_c)
            spk = str(row[spk_c]) if spk_c else uid.split("_")[0]
            chap = str(row[chap_c]) if chap_c else uid.split("_")[1]
            text = str(row[text_c]).strip()
            if not text:
                continue
            records.append(
                {
                    "speaker": spk,
                    "chapter": chap,
                    "idx": _idx_from_uid(uid),
                    "text": text,
                    "wav": uid,
                }
            )
            audio[uid] = row[audio_c]
    return records, audio


def _decode(cell):
    """Decode one HF audio cell → (np.float array, sr). Handles {bytes}/{array}."""
    import numpy as np
    import soundfile as sf

    if isinstance(cell, dict):
        if cell.get("bytes") is not None:
            wav, sr = sf.read(io.BytesIO(cell["bytes"]))
        elif cell.get("array") is not None:
            wav, sr = np.asarray(cell["array"]), int(cell.get("sampling_rate", 24000))
        else:  # {'path': ...}
            wav, sr = sf.read(cell["path"])
    else:
        wav, sr = (
            sf.read(io.BytesIO(cell)) if isinstance(cell, bytes) else sf.read(cell)
        )
    if getattr(wav, "ndim", 1) > 1:
        wav = wav.mean(axis=1)
    return wav, sr


def _write_concat(uids: list[str], audio: dict, out_path: str) -> None:
    import numpy as np
    import soundfile as sf

    chunks, sr = [], None
    for uid in uids:
        wav, this_sr = _decode(audio[uid])
        sr = sr or this_sr
        chunks.append(wav)
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
    refs_dir = os.path.join(out_dir, "refs")
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(refs_dir, exist_ok=True)

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    records, audio = load_libritts_rows(libritts_root)
    print(f"collected {len(records)} LibriTTS-R utterances from parquet")
    plan = plan_long_concatenations(records, buckets=LONG_BUCKETS)

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
            # Full paths to WRITE the audio; manifest stores paths RELATIVE to the
            # libritts_long root (gt/…, refs/…) because the loader re-joins them
            # against that root — storing the prefixed path double-prefixes it.
            gt_rel = os.path.join("gt", f"long_{b}_{n}.wav")
            ref_rel = os.path.join("refs", f"long_{b}_{n}_ref.wav")
            _write_concat(rec["member_wavs"], audio, os.path.join(out_dir, gt_rel))
            _write_concat([rec["ref_wav"]], audio, os.path.join(out_dir, ref_rel))
            # seed-tts-style row: id|ref_text|ref_wav|target_text|gt_wav.
            fh.write(
                f"long_{b}_{n}|{rec['ref_text']}|{ref_rel}|"
                f"{rec['target_text']}|{gt_rel}\n"
            )
            written += 1

    print(f"wrote {written} long passages → {manifest}")
    for b in sorted(per_bucket):
        print(f"  bucket {b}: {per_bucket[b]} passages")


if __name__ == "__main__":
    main()
