"""Validate the evaluation text set BEFORE spending the full sweep on it.

Answers "is this good eval text?" with data, not assertion (see the plan's
criteria). For every candidate sentence it runs ONE baseline (uncompressed)
generation and reports:

  - baseline CER — flags ``floor`` (==0, no headroom for compression to show) and
    ``unsynth`` (CER above a ceiling: the model fails it even uncompressed, which
    would confound compression with un-synthesizability).
  - decode length (talker tokens) — a per-group histogram, so the long-context
    group is confirmed to actually stress the cache.
  - reference-clip cleanliness — ASR the prompt / ground-truth clip and flag ones
    whose own transcription diverges (a dirty reference poisons speaker similarity).

Writes ``results/eval_set_report.md`` (keep/drop table + histogram) — the artifact
that *defines* the kept set. Needs the model + metrics (GPU recommended).

Run: ``make validate-eval`` or
``python tools/validate_eval_set.py --device cuda --data-dir data``.
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_QWEN_BENCH = os.path.join(_REPO, "models", "Qwen3-TTS", "benchmarks")
if _QWEN_BENCH not in sys.path:
    sys.path.insert(0, _QWEN_BENCH)

from turboquant.bench_common import sentence_hash
from turboquant.eval_sentences import (
    available_groups,
    iter_eval_items,
    predict_tokens,
)


def _estimate_snr_db(wav, sr) -> float:
    """Rough frame-energy SNR (dB): top-half-energy speech vs 10th-pct noise floor.

    No VAD — a cheap, monotone proxy good enough to flag clearly-noisy reference
    clips. Higher = cleaner. Returns -inf-safe finite numbers.
    """
    import numpy as np

    frame = max(1, int(0.02 * sr))  # 20 ms frames
    n = len(wav) // frame
    if n < 4:
        return 0.0
    energies = np.array(
        [np.mean(wav[i * frame : (i + 1) * frame] ** 2) for i in range(n)]
    )
    eps = float(np.mean(energies)) * 1e-6 + 1e-12
    # Peak-to-floor ratio: 95th-pct frame (speech) vs 5th-pct frame (silence/noise
    # floor). Relies on the clip containing both speech and pauses — true for the
    # Common Voice / seedtts prompts. A monotone cleanliness proxy, not a calibrated
    # SNR; the threshold (--min-ref-snr-db) is what flags clearly-noisy clips.
    noise = np.percentile(energies, 5) + eps
    speech = np.percentile(energies, 95) + eps
    return float(10.0 * np.log10(speech / noise))


def _evaluate_item(model, metrics, speaker, item, device):
    """One baseline generation → (cer, n_ar_tokens, ref_cer, ref_snr_db)."""
    from benchmark_qwen3tts_real import run_generation  # heavy import, deferred

    import librosa

    wavs, sr, _, _, _, n_ar_tokens = run_generation(
        model,
        item.text,
        "English",
        speaker,
        None,  # baseline / uncompressed
        device=device,
        ref_audio=item.ref_audio,
        ref_text=item.ref_text,
    )
    cer, _ = metrics.whisper_cer(wavs[0], sr, item.text)

    ref_cer = None
    ref_snr = None
    ref_path = item.ground_truth_audio or item.ref_audio
    ref_text = item.text if item.ground_truth_audio else item.ref_text
    if ref_path:
        try:
            rw, rsr = librosa.load(ref_path, sr=None, mono=True)
            ref_snr = _estimate_snr_db(rw, rsr)
            if ref_text:
                ref_cer, _ = metrics.whisper_cer(rw, rsr, ref_text)
        except Exception:
            ref_cer, ref_snr = None, None
    return cer, int(n_ar_tokens), ref_cer, ref_snr


def _ref_clean(ref_cer, ref_snr, ceiling: float, min_snr_db: float):
    """Is the reference clip clean enough to trust for speaker similarity?

    Returns True/False, or None when there is no reference to assess (curated
    cells carry no own clip — they clone from the shared --default-ref-audio).
    """
    if ref_cer is None and ref_snr is None:
        return None
    cer_ok = ref_cer is None or ref_cer <= ceiling
    snr_ok = ref_snr is None or ref_snr >= min_snr_db
    return bool(cer_ok and snr_ok)


def _verdict(cer: float, clean, ceiling: float) -> str:
    """keep / floor / unsynth / dirty-ref for one sentence."""
    if cer > ceiling:
        return "unsynth"
    if cer == 0.0:
        return "floor"
    if clean is False:
        return "dirty-ref"
    return "keep"


def _histogram(lengths: list[int], bins: int = 5) -> list[str]:
    """Text histogram of decode lengths (token counts)."""
    if not lengths:
        return ["  (no lengths)"]
    lo, hi = min(lengths), max(lengths)
    if lo == hi:
        return [f"  all {len(lengths)} sentences at {lo} tokens"]
    width = (hi - lo) / bins
    rows = []
    for b in range(bins):
        left = lo + b * width
        right = lo + (b + 1) * width
        count = sum(
            1 for x in lengths if (left <= x < right or (b == bins - 1 and x == hi))
        )
        rows.append(f"  {int(left):>5}-{int(right):>5} tok | {'#' * count} {count}")
    return rows


def _render(results: list[dict], lines: list[str]) -> None:
    """keep/drop table + length-heuristic check + per-group histogram into ``lines``."""
    lines.append("# Eval-set validation report\n")
    lines.append(
        "| group | sentence | CER | pred|act tok | refCER | refSNR | clean | verdict |"
    )
    lines.append("|---|---|---:|---:|---:|---:|:--:|---|")
    for r in results:
        preview = r["text"][:44].replace("|", "/")
        ref = "---" if r["ref_cer"] is None else f"{r['ref_cer']:.3f}"
        snr = "---" if r["ref_snr"] is None else f"{r['ref_snr']:.0f}"
        clean = {True: "Y", False: "N", None: "-"}[r["clean"]]
        lines.append(
            f"| {r['group']} | {preview} | {r['cer']:.3f} | "
            f"{r['pred_tok']}|{r['n_tok']} | {ref} | {snr} | {clean} | {r['verdict']} |"
        )
    kept = sum(1 for r in results if r["verdict"] == "keep")
    lines.append(f"\n**{kept}/{len(results)} sentences kept.**\n")

    # Heuristic accuracy: predicted vs actual generated tokens (median abs % error).
    errs = [
        abs(r["pred_tok"] - r["n_tok"]) / r["n_tok"] for r in results if r["n_tok"] > 0
    ]
    if errs:
        med = sorted(errs)[len(errs) // 2]
        lines.append(
            f"Length heuristic (≈6.4×words): median |predicted−actual|/actual = "
            f"{med:.1%} over {len(errs)} sentences.\n"
        )

    lines.append("## Decode-length histogram per group\n")
    for group in sorted({r["group"] for r in results}):
        lines.append(f"\n### {group}")
        lines += _histogram([r["n_tok"] for r in results if r["group"] == group])


def _write_ref_quality_csv(results: list[dict], path: str) -> int:
    """Sidecar (sentence_hash → ref_cer, ref_snr, clean) for analyze to join.

    Only rows that actually have a reference clip are written. Returns the count.
    """
    import csv

    rows = [r for r in results if r["ref_cer"] is not None or r["ref_snr"] is not None]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["sentence_hash", "group", "ref_cer", "ref_snr_db", "clean"])
        for r in rows:
            writer.writerow(
                [
                    r["sentence_hash"],
                    r["group"],
                    "" if r["ref_cer"] is None else f"{r['ref_cer']:.4f}",
                    "" if r["ref_snr"] is None else f"{r['ref_snr']:.2f}",
                    "" if r["clean"] is None else int(r["clean"]),
                ]
            )
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--metrics-device", default="cpu")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--groups", default=",".join(available_groups()))
    parser.add_argument("--max-per-group", type=int, default=None)
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    parser.add_argument(
        "--cer-ceiling",
        type=float,
        default=0.3,
        help="Baseline CER above which a sentence is dropped as un-synthesizable.",
    )
    parser.add_argument(
        "--min-ref-snr-db",
        type=float,
        default=15.0,
        help="Reference clips below this estimated SNR are flagged noisy (clean=N).",
    )
    parser.add_argument("--out-md", default="results/eval_set_report.md")
    parser.add_argument(
        "--ref-quality-out",
        default="results/ref_quality.csv",
        help="Sidecar CSV (sentence_hash → ref_cer, ref_snr, clean) for analyze.",
    )
    args = parser.parse_args()

    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    items = iter_eval_items(groups, args.max_per_group, args.data_dir)
    print(f"Validating {len(items)} sentences across {groups} ...")

    from benchmark_qwen3tts_real import QualityMetrics
    from qwen_tts import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(args.model, device_map=args.device)
    speakers = model.get_supported_speakers()
    speaker = speakers[0] if speakers else "Ryan"
    metrics = QualityMetrics(device=args.metrics_device)

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    results = []
    iterator = enumerate(items)
    if tqdm is not None:
        iterator = tqdm(iterator, total=len(items), desc="validate", unit="item")
    for i, item in iterator:
        clean = None
        try:
            cer, n_tok, ref_cer, ref_snr = _evaluate_item(
                model, metrics, speaker, item, args.device
            )
            clean = _ref_clean(ref_cer, ref_snr, args.cer_ceiling, args.min_ref_snr_db)
            verdict = _verdict(cer, clean, args.cer_ceiling)
        except Exception as exc:  # a generation failure is itself a drop signal
            cer, n_tok, ref_cer, ref_snr, verdict = 1.0, 0, None, None, f"error:{exc}"
        results.append(
            {
                "group": item.group,
                "text": item.text,
                "sentence_hash": sentence_hash(item.text),
                "cer": cer,
                "n_tok": n_tok,
                "pred_tok": predict_tokens(item.text),
                "ref_cer": ref_cer,
                "ref_snr": ref_snr,
                "clean": clean,
                "verdict": verdict,
            }
        )
        msg = f"  [{i + 1}/{len(items)}] {item.group} CER={cer:.3f} {verdict}"
        tqdm.write(msg) if tqdm is not None else print(msg)

    n_ref = _write_ref_quality_csv(results, args.ref_quality_out)
    print(f"Wrote {n_ref} reference-quality rows → {args.ref_quality_out}")

    lines: list[str] = []
    _render(results, lines)
    os.makedirs(os.path.dirname(args.out_md) or ".", exist_ok=True)
    with open(args.out_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {args.out_md}")


if __name__ == "__main__":
    main()
