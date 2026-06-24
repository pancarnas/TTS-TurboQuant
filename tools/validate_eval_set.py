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

from turboquant.eval_sentences import available_groups, iter_eval_items


def _evaluate_item(model, metrics, speaker, item, device):
    """One baseline generation → (cer, n_ar_tokens, ref_cer or None)."""
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
    ref_path = item.ground_truth_audio or item.ref_audio
    ref_text = item.text if item.ground_truth_audio else item.ref_text
    if ref_path and ref_text:
        try:
            rw, rsr = librosa.load(ref_path, sr=None, mono=True)
            ref_cer, _ = metrics.whisper_cer(rw, rsr, ref_text)
        except Exception:
            ref_cer = None
    return cer, int(n_ar_tokens), ref_cer


def _verdict(cer: float, ref_cer, ceiling: float) -> str:
    """keep / floor / unsynth / dirty-ref for one sentence."""
    if cer > ceiling:
        return "unsynth"
    if cer == 0.0:
        return "floor"
    if ref_cer is not None and ref_cer > ceiling:
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
    """keep/drop table + per-group length histogram into ``lines``."""
    lines.append("# Eval-set validation report\n")
    lines.append("| group | sentence | CER | len(tok) | refCER | verdict |")
    lines.append("|---|---|---:|---:|---:|---|")
    for r in results:
        preview = r["text"][:48].replace("|", "/")
        ref = "---" if r["ref_cer"] is None else f"{r['ref_cer']:.3f}"
        lines.append(
            f"| {r['group']} | {preview} | {r['cer']:.3f} | {r['n_tok']} | "
            f"{ref} | {r['verdict']} |"
        )
    kept = sum(1 for r in results if r["verdict"] == "keep")
    lines.append(f"\n**{kept}/{len(results)} sentences kept.**\n")

    lines.append("## Decode-length histogram per group\n")
    for group in sorted({r["group"] for r in results}):
        lines.append(f"\n### {group}")
        lines += _histogram([r["n_tok"] for r in results if r["group"] == group])


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
    parser.add_argument("--out-md", default="results/eval_set_report.md")
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

    results = []
    for i, item in enumerate(items):
        try:
            cer, n_tok, ref_cer = _evaluate_item(
                model, metrics, speaker, item, args.device
            )
            verdict = _verdict(cer, ref_cer, args.cer_ceiling)
        except Exception as exc:  # a generation failure is itself a drop signal
            cer, n_tok, ref_cer, verdict = 1.0, 0, None, f"error:{exc}"
        results.append(
            {
                "group": item.group,
                "text": item.text,
                "cer": cer,
                "n_tok": n_tok,
                "ref_cer": ref_cer,
                "verdict": verdict,
            }
        )
        print(f"  [{i + 1}/{len(items)}] {item.group} CER={cer:.3f} {verdict}")

    lines: list[str] = []
    _render(results, lines)
    os.makedirs(os.path.dirname(args.out_md) or ".", exist_ok=True)
    with open(args.out_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {args.out_md}")


if __name__ == "__main__":
    main()
