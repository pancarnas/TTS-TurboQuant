"""Teacher-forced PPL + KL + attention-divergence diagnostic for VALL-E-X.

For each eval sentence with ground-truth audio, EnCodec-tokenize the real
recording and run the AR decoder teacher-forced over its codebook-0 tokens —
once at fp16, once per quantized config. No sampling, no vocoder, no ASR:
this measures the model's predictive distribution directly, the same protocol
as the LLM KV-quant literature's perplexity tables.

Per (sentence, config) row (--out):
  nll / ppl            — teacher-forced NLL / perplexity on GT tokens
  delta_nll            — nll minus the same sentence's fp16 nll
  kl_mean / kl_p95     — KL(fp16 || quantized) over per-step distributions
  top1_agree           — fraction of steps where both argmax the same token
  first_flip           — first step where argmax disagrees ('' = never)
  fp16_margin_*        — top1-top2 logit margin stats of the fp16 pass

Per (sentence, config, layer, strided-pos) row (--divergence-out): the same
attention-divergence metrics as the Qwen experiment (attn_js, attn_top1,
cos_k, ...), computed quantized-vs-exact inside the decoder via
DivergenceRecorder — so tools/analyze_divergence.py reads this CSV too.

Run from the repo root (PYTHONPATH per scripts/eddie/eddie_run.sh):
  python models/VALL-E-X/benchmarks/vallex_ppl_divergence.py \
      --groups seedtts_en,librispeech_pc --max-per-group 50 \
      --configs "K4V4@64,K4V3@64,K3V3@64,K3V3@128" --protected-layers 2 \
      --device cuda --out results/vallex_ppl.csv \
      --divergence-out results/vallex_attn_divergence.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_VALLEX_DIR = os.path.dirname(_THIS_DIR)
_REPO_ROOT = os.path.dirname(os.path.dirname(_VALLEX_DIR))
for _p in (_REPO_ROOT, _VALLEX_DIR, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from benchmark_vallex_real import (  # noqa: E402
    load_vallex_model,
    prepare_text_and_prompt,
)
from data.tokenizer import tokenize_audio  # noqa: E402
from turboquant.config import TurboQuantConfig  # noqa: E402
from turboquant.eval_sentences import iter_eval_items  # noqa: E402
from turboquant_cache import DivergenceRecorder  # noqa: E402

_CFG_RE = re.compile(r"[Kk](\d+)[Vv](\d+)@(\d+)$")

PPL_COLUMNS = [
    "group",
    "idx",
    "config",
    "key_bits",
    "value_bits",
    "rw",
    "protected_layers",
    "n_tokens",
    "nll",
    "ppl",
    "delta_nll",
    "kl_mean",
    "kl_p95",
    "top1_agree",
    "first_flip",
    "fp16_margin_mean",
    "fp16_margin_p10",
]

DIV_COLUMNS = [
    "group",
    "idx",
    "layer",
    "pos",
    "key_bits",
    "value_bits",
    "rw",
    "protected_layers",
] + DivergenceRecorder.METRICS


def parse_specs(spec: str) -> list[tuple[int, int, int]]:
    """'K4V4@64,K3V3@128' -> [(4,4,64), (3,3,128)]. AR-only (no stage prefixes
    here — NAR does not run in the teacher-forced AR loop)."""
    out = []
    for tok in (t.strip() for t in spec.split(",")):
        if not tok or tok.lower() == "fp16":  # fp16 reference always runs
            continue
        m = _CFG_RE.match(tok)
        if m is None:
            raise SystemExit(f"bad config token {tok!r}; expected e.g. K4V4@64")
        out.append((int(m.group(1)), int(m.group(2)), int(m.group(3))))
    if not out:
        raise SystemExit("--configs contained no quantized configs")
    return out


def gt_codebook0(codec, gt_path: str, max_tokens: int) -> torch.Tensor:
    """Codebook-0 EnCodec tokens of the ground-truth recording, (T,) long."""
    frames = tokenize_audio(codec, gt_path)
    codes = frames[0][0]  # (B, n_q, T)
    return codes[0, 0, :max_tokens].long()


def margin_stats(logits: torch.Tensor) -> tuple[float, float]:
    """(mean, p10) of the top1-top2 logit gap per step; logits (T, V)."""
    top2 = logits.topk(2, dim=-1).values
    margin = (top2[:, 0] - top2[:, 1]).float()
    return float(margin.mean()), float(margin.quantile(0.10))


def kl_and_flips(logits_ref: torch.Tensor, logits_q: torch.Tensor):
    """KL(ref||q) per step, top-1 agreement, first argmax flip ('' if none)."""
    logp_ref = F.log_softmax(logits_ref, dim=-1)
    logp_q = F.log_softmax(logits_q, dim=-1)
    kl = (logp_ref.exp() * (logp_ref - logp_q)).sum(-1)
    agree = logits_ref.argmax(-1) == logits_q.argmax(-1)
    flips = (~agree).nonzero()
    first_flip = int(flips[0]) if len(flips) else ""
    return kl, float(agree.float().mean()), first_flip


def done_keys(out_path: str) -> set:
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        return set()
    with open(out_path, newline="", encoding="utf-8") as fh:
        return {
            (row["group"], int(row["idx"]), row["config"])
            for row in csv.DictReader(fh)
        }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--groups", default="seedtts_en,librispeech_pc")
    ap.add_argument("--max-per-group", type=int, default=50)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--configs", default="K4V4@64,K4V3@64,K3V3@64,K3V3@128")
    ap.add_argument("--protected-layers", type=int, default=2)
    ap.add_argument(
        "--step-stride",
        type=int,
        default=4,
        help="Record attention divergence every Nth decode step.",
    )
    ap.add_argument(
        "--max-forced-tokens",
        type=int,
        default=1125,
        help="Cap the forced GT token sequence (1125 = 15 s at 75 Hz).",
    )
    ap.add_argument("--preset", default="alan.npz")
    ap.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    ap.add_argument("--out", default="results/vallex_ppl.csv")
    ap.add_argument(
        "--divergence-out", default="results/vallex_attn_divergence.csv"
    )
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    specs = parse_specs(args.configs)
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    preset_path = os.path.join(_VALLEX_DIR, "presets", args.preset)

    model, codec, _vocos = load_vallex_model(args.device)

    done = done_keys(args.out) if args.resume else set()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    append = args.resume and os.path.exists(args.out) and os.path.getsize(args.out) > 0
    ppl_fh = open(args.out, "a" if append else "w", newline="", encoding="utf-8")
    ppl_writer = csv.writer(ppl_fh)
    div_append = (
        args.resume
        and os.path.exists(args.divergence_out)
        and os.path.getsize(args.divergence_out) > 0
    )
    div_fh = open(
        args.divergence_out, "a" if div_append else "w", newline="", encoding="utf-8"
    )
    div_writer = csv.writer(div_fh)
    if not append:
        ppl_writer.writerow(PPL_COLUMNS)
    if not div_append:
        div_writer.writerow(DIV_COLUMNS)

    skipped_no_gt = 0
    for group in groups:
        items = iter_eval_items([group], args.max_per_group, args.data_dir)
        for idx, item in enumerate(items):
            gt = getattr(item, "ground_truth_audio", None)
            if not gt or not os.path.exists(gt):
                skipped_no_gt += 1
                continue
            todo = [
                (kb, vb, rw)
                for kb, vb, rw in specs
                if (group, idx, f"K{kb}V{vb}@{rw}") not in done
            ]
            need_fp16 = (group, idx, "fp16") not in done
            if not todo and not need_fp16:
                continue

            forced = gt_codebook0(codec, gt, args.max_forced_tokens)
            if forced.numel() < 8:
                print(f"  {group}#{idx}: GT too short after tokenize; skipped")
                continue
            (
                text_tokens,
                text_tokens_lens,
                audio_prompts,
                enroll_x_lens,
                lang_pr,
                langs,
            ) = prepare_text_and_prompt(item.text, "en", preset_path)
            common = dict(
                x=text_tokens.to(args.device),
                x_lens=text_tokens_lens.to(args.device),
                y=audio_prompts.to(args.device),
                enroll_x_lens=enroll_x_lens,
                forced_tokens=forced,
                prompt_language=lang_pr,
                text_language=langs,
            )

            lp_ref, logits_ref = model.teacher_forced_ar(
                **common, turboquant_config=None
            )
            nll_ref = float(-lp_ref.mean())
            m_mean, m_p10 = margin_stats(logits_ref)
            if need_fp16:
                ppl_writer.writerow(
                    [group, idx, "fp16", "", "", "", "", forced.numel(),
                     round(nll_ref, 6), round(math.exp(nll_ref), 4), 0.0,
                     0.0, 0.0, 1.0, "", round(m_mean, 4), round(m_p10, 4)]
                )
                ppl_fh.flush()

            for kb, vb, rw in todo:
                cfg = TurboQuantConfig(
                    key_bits=kb,
                    value_bits=vb,
                    residual_window=rw,
                    protected_layers=args.protected_layers,
                    track_only=False,
                )
                recorder = DivergenceRecorder(rw, args.step_stride)
                lp, logits = model.teacher_forced_ar(
                    **common, turboquant_config=cfg, recorder=recorder
                )
                nll = float(-lp.mean())
                kl, top1_agree, first_flip = kl_and_flips(logits_ref, logits)
                label = f"K{kb}V{vb}@{rw}"
                ppl_writer.writerow(
                    [group, idx, label, kb, vb, rw, args.protected_layers,
                     forced.numel(), round(nll, 6),
                     round(math.exp(nll), 4), round(nll - nll_ref, 6),
                     round(float(kl.mean()), 6),
                     round(float(kl.quantile(0.95)), 6),
                     round(top1_agree, 4), first_flip,
                     round(m_mean, 4), round(m_p10, 4)]
                )
                ppl_fh.flush()
                for row in recorder.rows:
                    div_writer.writerow(
                        [group, idx, row[0], row[1], kb, vb, rw,
                         args.protected_layers] + row[2:]
                    )
                div_fh.flush()
            print(
                f"{group}#{idx}: {forced.numel()} tokens, fp16 ppl "
                f"{math.exp(nll_ref):.2f}, {len(todo)} configs done"
            )

    ppl_fh.close()
    div_fh.close()
    if skipped_no_gt:
        print(f"note: {skipped_no_gt} items had no ground-truth audio; skipped")

    try:
        import pandas as pd

        df = pd.read_csv(args.out)
        print(f"\n== teacher-forced summary ({len(df)} rows) ==")
        print(
            df.groupby("config")[
                ["ppl", "delta_nll", "kl_mean", "top1_agree"]
            ].mean().round(4)
        )
    except Exception as exc:  # noqa: BLE001 - summary is best-effort
        print(f"(summary skipped: {exc})")


if __name__ == "__main__":
    main()
