"""Re-score generated audio with a larger Whisper model (default large-v3).

Why this exists: the in-benchmark CER fed the raw generated waveform to Whisper
**without resampling to 16 kHz**, so 24 kHz audio was heard ~1.5x sped-up — which
inflated CER, worst on the hard/long/collapsed tail. On top of that, Whisper
``base`` mis-transcribes rapid repetition (tongue-twisters), causing false
collapses. This standalone pass fixes both: it loads every saved wav, **resamples
to 16 kHz**, transcribes with a strong model (``large-v3``), and recomputes the
field-standard normalized CER — no GPU TTS, no regeneration, just re-scoring.

It reads each trials CSV, reconstructs the wav path the benchmark wrote (see
``benchmark_qwen3tts_real.py``: ``qwen_{group}_{idx}_{arm}_s{seed}_t{temp}_{safe}``),
re-scores it, and writes ``<stem>_rescored_<model>.csv`` — the SAME columns with
``cer`` replaced by the corrected value (so ``analyze_kv_benchmark.py`` runs on it
unchanged), plus ``cer_orig`` and ``transcript`` for audit.

Run on the box (GPU recommended for large-v3):
  python tools/rescore_audio.py \
      --trials-glob 'results/qwen_trials_shard*_clone_rw24_t09_*.csv' \
      --data-dir data --model large-v3 --device cuda
Then re-run analyze_kv_benchmark.py with --trials-glob '...rescored_large-v3*.csv'.
"""

from __future__ import annotations

import argparse
import glob
import math
import os

import numpy as np
import pandas as pd

from turboquant.eval_sentences import iter_eval_items

DEFAULT_AUDIO_DIR = "models/Qwen3-TTS/benchmarks/outputs"
DEFAULT_MODEL = "large-v3"
COLLAPSE_CER = 0.5


def wav_name(row: pd.Series, prefix: str = "qwen") -> str:
    """Reconstruct the benchmark's saved-wav filename for one trial row."""
    safe = str(row["config"]).replace(" ", "_").replace("/", "_")
    temp = row.get("temperature", None)
    tsuf = "" if pd.isna(temp) else f"_t{temp}"
    return (
        f"{prefix}_{row['group']}_{int(row['idx'])}_{row['arm']}"
        f"_s{int(row['seed'])}{tsuf}_{safe}.wav"
    )


def resolve_wav(row: pd.Series, audio_dir: str, prefix: str = "qwen") -> str | None:
    """Exact path, with a temperature-format-tolerant glob fallback."""
    exact = os.path.join(audio_dir, wav_name(row, prefix))
    if os.path.exists(exact):
        return exact
    safe = str(row["config"]).replace(" ", "_").replace("/", "_")
    pat = os.path.join(
        audio_dir,
        f"{prefix}_{row['group']}_{int(row['idx'])}_{row['arm']}"
        f"_s{int(row['seed'])}_t*_{safe}.wav",
    )
    hits = glob.glob(pat)
    return hits[0] if hits else None


def to_16k(wav: np.ndarray, sr: int) -> np.ndarray:
    """Mono float32 resampled to 16 kHz (what Whisper actually expects)."""
    from scipy.signal import resample_poly

    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)
    if sr == 16000:
        return wav
    g = math.gcd(16000, sr)
    return resample_poly(wav, 16000 // g, sr // g).astype(np.float32)


def build_text_lookup(groups, max_per_group, data_dir: str) -> dict:
    """(group, idx) -> target text, matching the benchmark's item ordering."""
    lookup: dict[tuple[str, int], str] = {}
    for g in groups:
        try:
            items = iter_eval_items([g], max_per_group, data_dir)
        except Exception as exc:  # noqa: BLE001 - report and skip unloadable group
            print(f"  (could not load group {g}: {exc})")
            continue
        for i, it in enumerate(items):
            lookup[(g, i)] = it.text
    return lookup


class CERScorer:
    """Whisper EnglishTextNormalizer + jiwer CER, applied to ref and hyp alike."""

    def __init__(self) -> None:
        from whisper.normalizers import EnglishTextNormalizer

        self._norm = EnglishTextNormalizer()

    def __call__(self, target: str, hypothesis: str) -> float:
        from jiwer import cer

        ref = self._norm(target or "")
        hyp = self._norm(hypothesis or "")
        return float(cer(ref, hyp)) if ref else 0.0


class WhisperTranscriber:
    """Loads a Whisper model once; transcribes a wav path at a forced language."""

    def __init__(self, model: str, device: str, language: str = "en") -> None:
        import whisper

        self._model = whisper.load_model(model, device=device)
        self._language = language

    def __call__(self, path: str) -> str:
        import soundfile as sf

        wav, sr = sf.read(path)
        result = self._model.transcribe(to_16k(wav, sr), language=self._language)
        return str(result.get("text", "")).strip()


def score_trials(df, text_lookup, audio_dir, prefix, transcribe, score) -> pd.DataFrame:
    """Add corrected ``cer`` (keeping ``cer_orig``) + ``transcript`` per row.

    ``transcribe(path) -> str`` and ``score(target, hyp) -> float`` are injected so
    the orchestration is testable without loading Whisper.
    """
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover - tqdm optional
        tqdm = None

    new_cer, transcripts, missing = [], [], 0
    rows = df.iterrows()
    it = tqdm(rows, total=len(df), desc="rescore", unit="wav") if tqdm else rows
    for _, row in it:
        path = resolve_wav(row, audio_dir, prefix)
        if path is None:
            new_cer.append(np.nan)
            transcripts.append("")
            missing += 1
            continue
        text = text_lookup.get((row["group"], int(row["idx"])), "")
        hyp = transcribe(path)
        new_cer.append(score(text, hyp))
        transcripts.append(hyp)
    out = df.copy()
    out["cer_orig"] = out["cer"]
    out["cer"] = new_cer
    out["transcript"] = transcripts
    if missing:
        print(f"  WARNING: {missing}/{len(df)} wavs not found on disk")
    return out


def summarize(df: pd.DataFrame) -> None:
    """Per-config corrected mean CER + collapse rate vs the original."""
    valid = df.dropna(subset=["cer"])
    print(f"\n== corrected scores ({len(valid)} trials) ==")
    for cfg, g in valid.groupby("config"):
        coll = (g["cer"] > COLLAPSE_CER).mean()
        coll0 = (g["cer_orig"] > COLLAPSE_CER).mean()
        print(
            f"{cfg:22s} n={len(g):4d}  collapse {coll0:5.1%}->{coll:5.1%}  "
            f"meanCER {g['cer_orig'].mean():.3f}->{g['cer'].mean():.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials-glob", required=True)
    parser.add_argument("--audio-dir", default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None, help="cuda/cpu (default: auto)")
    parser.add_argument("--language", default="en")
    parser.add_argument("--prefix", default="qwen")
    parser.add_argument("--max-per-group", type=int, default=None)
    parser.add_argument("--out-suffix", default=None)
    args = parser.parse_args()

    device = args.device
    if device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    suffix = args.out_suffix or f"_rescored_{args.model}"

    paths = sorted(glob.glob(args.trials_glob))
    if not paths:
        raise SystemExit(f"no CSVs match {args.trials_glob}")
    frames = {p: pd.read_csv(p) for p in paths}
    groups = sorted({g for df in frames.values() for g in df["group"].unique()})
    print(f"model={args.model} device={device} groups={groups}")

    text_lookup = build_text_lookup(groups, args.max_per_group, args.data_dir)
    transcribe = WhisperTranscriber(args.model, device, args.language)
    score = CERScorer()

    combined = []
    for path, df in frames.items():
        out = score_trials(
            df, text_lookup, args.audio_dir, args.prefix, transcribe, score
        )
        out_path = path.replace(".csv", f"{suffix}.csv")
        out.to_csv(out_path, index=False)
        print(f"wrote {out_path}")
        combined.append(out)
    summarize(pd.concat(combined, ignore_index=True))


if __name__ == "__main__":
    main()
