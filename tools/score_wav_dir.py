"""Score experiment wavs offline: Whisper CER + WER + WavLM speaker sim vs fp16.

The generation benchmarks save one wav per (sentence, config) and — when run
with metrics disabled (``--no-quality`` on VALL-E-X, or the Qwen divergence
experiment, which never computes perceptual metrics) — leave scoring to this
standalone pass:

  * CER + WER: Whisper (default large-v3, resampled to 16 kHz) against the
    ground-truth text for EVERY config; both use the EnglishTextNormalizer on
    ref and hyp alike (tools/rescore_audio.py scorers);
  * SpkSim: WavLM x-vector cosine of each compressed wav against the SAME
    sentence's uncompressed baseline wav (the ``fp16`` config, generated under
    the same arm/seed/temperature). Baseline rows get CER/WER but a blank
    spk_sim — cosine against itself is trivially 1.

Recognized filenames (anything else in the dir is counted and skipped):
  qwen_{group}_{idx}_sampling_s{seed}_t{temp}_K{kb}_V{vb}_rw={rw}_pl={pl}.wav
  qwen_{group}_{idx}_sampling_s{seed}_t{temp}_fp16.wav
  vallex_{group}_{idx}_{arm}_s{seed}[_t{temp}]_{config}.wav
    (config = fp16 | K{kb}V{vb}@{rw}[-nar|-both]; no pl in VALL-E-X names —
    protection is global per run, the ``pl`` column is -1 there)

Rows stream to --out per wav (crash-safe); --resume skips already-scored rows.

Run on the box, after the fp16 baseline wavs exist:
  python tools/score_wav_dir.py \
      --audio-dir models/Qwen3-TTS/benchmarks/outputs \
      --data-dir data --device cuda --out results/wav_scores.csv
  python tools/score_wav_dir.py \
      --audio-dir models/VALL-E-X/benchmarks/outputs \
      --data-dir data --device cuda --out results/vallex_wav_scores.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_AUDIO_DIR = "models/Qwen3-TTS/benchmarks/outputs"
BASELINE = "fp16"
COLLAPSE_CER = 0.5

COLUMNS = [
    "group",
    "idx",
    "arm",
    "seed",
    "temperature",
    "config",
    "key_bits",
    "value_bits",
    "rw",
    "pl",
    "dur_s",
    "cer",
    "wer",
    "spk_sim",
    "transcript",
    "wav",
]

# fp16 rows have no protection setting; -1 keeps the resume key uniformly int.
# VALL-E-X rows use it for every config (pl is global per run, not in names).
FP16_PL = -1

_WAV_RE = re.compile(
    r"^qwen_(?P<group>.+)_(?P<idx>\d+)_sampling_s(?P<seed>\d+)_t(?P<temp>[0-9.]+)"
    r"_(?:K(?P<kb>\d+)_V(?P<vb>\d+)_rw=(?P<rw>\d+)_pl=(?P<pl>\d+)|fp16)\.wav$"
)

# VALL-E-X benchmark naming (benchmark_vallex_real.py --configs runs). The
# optional -nar/-both suffix marks which decoder stage was quantized.
_VALLEX_WAV_RE = re.compile(
    r"^vallex_(?P<group>.+)_(?P<idx>\d+)_(?P<arm>[a-z]+)_s(?P<seed>\d+)"
    r"(?:_t(?P<temp>[0-9.]+))?"
    r"_(?:K(?P<kb>\d+)V(?P<vb>\d+)@(?P<rw>\d+)(?P<stage>-nar|-both)?|fp16)\.wav$"
)


def parse_wav_name(name: str) -> dict | None:
    """Fields of one experiment wav filename, or None if foreign.

    The greedy ``group`` pattern still splits correctly because ``idx`` is the
    LAST ``_<digits>_<arm>`` run before the fixed suffix, and group names
    (seedtts_en, librispeech_pc, ...) never end in ``_<digits>``.
    """
    m = _WAV_RE.match(name)
    if m:
        d = m.groupdict()
        fp16 = d["kb"] is None
        return {
            "group": d["group"],
            "idx": int(d["idx"]),
            "seed": int(d["seed"]),
            "temperature": float(d["temp"]),
            "key_bits": "" if fp16 else int(d["kb"]),
            "value_bits": "" if fp16 else int(d["vb"]),
            "rw": "" if fp16 else int(d["rw"]),
            "pl": FP16_PL if fp16 else int(d["pl"]),
            "config": BASELINE if fp16 else f"K{d['kb']}V{d['vb']}@{d['rw']}",
            "wav": name,
        }
    m = _VALLEX_WAV_RE.match(name)
    if m:
        d = m.groupdict()
        fp16 = d["kb"] is None
        config = (
            BASELINE
            if fp16
            else f"K{d['kb']}V{d['vb']}@{d['rw']}" + (d["stage"] or "")
        )
        return {
            "group": d["group"],
            "idx": int(d["idx"]),
            "arm": d["arm"],
            "seed": int(d["seed"]),
            "temperature": float(d["temp"]) if d["temp"] is not None else "",
            "key_bits": "" if fp16 else int(d["kb"]),
            "value_bits": "" if fp16 else int(d["vb"]),
            "rw": "" if fp16 else int(d["rw"]),
            "pl": FP16_PL,
            "config": config,
            "wav": name,
        }
    return None


def collect_entries(audio_dir: str) -> tuple[list[dict], int]:
    """(parsed entries, foreign-file count) for ``audio_dir``, sorted so all
    configs of one sentence are adjacent — scoring then reuses the baseline
    embedding without a cache spanning the whole run."""
    out, foreign = [], 0
    for name in sorted(os.listdir(audio_dir)):
        if not name.endswith(".wav"):
            continue
        e = parse_wav_name(name)
        if e is None:
            foreign += 1
            continue
        e["path"] = os.path.join(audio_dir, name)
        out.append(e)
    out.sort(
        key=lambda e: (
            e["group"],
            e["idx"],
            e.get("arm", ""),
            e["seed"],
            str(e["temperature"]),
            e["config"],
        )
    )
    return out, foreign


def _sentence_key(e: dict) -> tuple:
    """One generated rendition: (group, idx, arm, seed, temperature)."""
    return (e["group"], e["idx"], e.get("arm", ""), e["seed"], str(e["temperature"]))


def attach_baseline(entries: list[dict]) -> int:
    """Set ``baseline_path`` on every entry (None on fp16 rows themselves and on
    sentences whose fp16 wav is missing). Returns the missing count so the
    caller can warn — a silent None would just look like an unscored column."""
    base = {_sentence_key(e): e["path"] for e in entries if e["config"] == BASELINE}
    missing = 0
    for e in entries:
        if e["config"] == BASELINE:
            e["baseline_path"] = None
            continue
        e["baseline_path"] = base.get(_sentence_key(e))
        if e["baseline_path"] is None:
            missing += 1
    return missing


def entry_key(e: dict) -> tuple:
    return (e["group"], e["idx"], e.get("arm", ""), e["seed"], e["config"], e["pl"])


def done_keys(out_path: str) -> set:
    """(group, idx, arm, seed, config, pl) already scored — for --resume.

    ``arm``/``seed`` fall back gracefully so CSVs written before those columns
    existed still resume (arm defaults to '', matching Qwen entries)."""
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        return set()
    done = set()
    with open(out_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            done.add(
                (
                    row["group"],
                    int(row["idx"]),
                    row.get("arm", "") or "",
                    int(row.get("seed", 0) or 0),
                    row["config"],
                    int(row["pl"]),
                )
            )
    return done


def score_entries(
    entries: list[dict],
    text_lookup: dict,
    transcribe,
    score,
    embed,
    wav_duration,
    emit,
    score_wer=None,
) -> None:
    """Score each entry and hand the finished row dict to ``emit``.

    ``transcribe(path)->str``, ``score(ref, hyp)->float``, ``embed(path)->vector``
    (normalized), and ``wav_duration(path)->float`` are injected so this
    orchestration is testable without Whisper/WavLM/soundfile. ``score_wer``
    (optional, same signature as ``score``) fills the ``wer`` column — blank
    when not given. Baseline embeddings are computed once per sentence
    (entries are sentence-sorted).
    """
    emb_cache: dict[str, object] = {}

    def cached_embed(path):
        if path not in emb_cache:
            emb_cache.clear()  # sentence-sorted: old baseline is never needed again
            emb_cache[path] = embed(path)
        return emb_cache[path]

    for e in entries:
        text = text_lookup.get((e["group"], e["idx"]))
        if text is None:
            print(f"  WARNING: no ground-truth text for {e['wav']}; skipped")
            continue
        hyp = transcribe(e["path"])
        spk_sim = ""
        if e["baseline_path"]:
            a, b = cached_embed(e["baseline_path"]), embed(e["path"])
            spk_sim = float(sum(x * y for x, y in zip(a, b)))
        emit(
            {
                **{k: e[k] for k in COLUMNS if k in e},
                "dur_s": wav_duration(e["path"]),
                "cer": score(text, hyp),
                "wer": score_wer(text, hyp) if score_wer is not None else "",
                "spk_sim": spk_sim,
                "transcript": hyp,
            }
        )


def summarize(out_path: str) -> None:
    """Per-config mean/collapse CER and mean SpkSim over the finished CSV."""
    import pandas as pd

    df = pd.read_csv(out_path)
    if df.empty:
        print("no rows scored")
        return
    print(f"\n== scores ({len(df)} wavs) ==")
    for cfg, g in df.groupby("config"):
        sim = g["spk_sim"].dropna()
        sim_txt = f"spkSim {sim.mean():.4f}" if len(sim) else "spkSim    --"
        wer_txt = ""
        if "wer" in g.columns:
            wer = g["wer"].dropna()
            if len(wer):
                wer_txt = f"  meanWER {wer.mean():.3f}"
        print(
            f"{cfg:14s} n={len(g):4d}  meanCER {g['cer'].mean():.3f}{wer_txt}  "
            f"collapse {(g['cer'] > COLLAPSE_CER).mean():5.1%}  {sim_txt}"
        )


class WavLMEmbedder:
    """WavLM x-vector speaker embeddings (normalized) — the same model as the
    in-benchmark QualityMetrics, but without importing the TTS stack."""

    def __init__(self, device: str) -> None:
        import torch
        from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

        self._torch = torch
        self._extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "microsoft/wavlm-base-plus-sv"
        )
        self._model = (
            WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv")
            .to(device)
            .eval()
        )
        self._device = device

    def __call__(self, path: str):
        import soundfile as sf

        from tools.rescore_audio import to_16k

        wav, sr = sf.read(path)
        inputs = self._extractor(
            to_16k(wav, sr), sampling_rate=16000, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with self._torch.no_grad():
            emb = self._model(**inputs).embeddings
            emb = self._torch.nn.functional.normalize(emb, dim=-1)
        return emb.squeeze().cpu().numpy()


def _wav_duration(path: str) -> float:
    import soundfile as sf

    info = sf.info(path)
    return round(info.frames / info.samplerate, 3) if info.samplerate else float("nan")


def main() -> None:
    from tools.rescore_audio import (
        CERScorer,
        WERScorer,
        WhisperTranscriber,
        build_text_lookup,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model", default="large-v3", help="Whisper model for CER.")
    parser.add_argument("--device", default=None, help="cuda/cpu (default: auto)")
    parser.add_argument("--language", default="en")
    parser.add_argument("--out", default="results/wav_scores.csv")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    device = args.device
    if device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    entries, foreign = collect_entries(args.audio_dir)
    if foreign:
        print(f"note: {foreign} wavs in {args.audio_dir} have foreign names; skipped")
    if not entries:
        raise SystemExit(f"no divergence-experiment wavs in {args.audio_dir}")
    missing = attach_baseline(entries)
    if missing:
        print(
            f"WARNING: {missing}/{len(entries)} wavs have no {BASELINE} baseline "
            "wav — their spk_sim will be blank"
        )
    if args.resume:
        done = done_keys(args.out)
        n0 = len(entries)
        entries = [e for e in entries if entry_key(e) not in done]
        print(f"resume: skipping {n0 - len(entries)} done, scoring {len(entries)}")

    groups = sorted({e["group"] for e in entries})
    print(f"model={args.model} device={device} groups={groups} n={len(entries)}")
    text_lookup = build_text_lookup(groups, None, args.data_dir)
    transcribe = WhisperTranscriber(args.model, device, args.language)
    score = CERScorer()
    score_wer = WERScorer()
    embed = WavLMEmbedder(device)

    try:
        from tqdm import tqdm

        bar = tqdm(total=len(entries), desc="score", unit="wav")
    except ImportError:  # pragma: no cover - tqdm optional
        bar = None

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    append = args.resume and os.path.exists(args.out) and os.path.getsize(args.out) > 0
    with open(args.out, "a" if append else "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        if not append:
            writer.writeheader()

        def emit(row: dict) -> None:
            writer.writerow(row)
            fh.flush()
            if bar:
                bar.update(1)

        score_entries(
            entries,
            text_lookup,
            transcribe,
            score,
            embed,
            _wav_duration,
            emit,
            score_wer=score_wer,
        )
    if bar:
        bar.close()
    summarize(args.out)


if __name__ == "__main__":
    main()
