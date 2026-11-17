"""Table-1 statistics (OmniKVQuant §2.2) for Qwen3-TTS talker keys/values, plus
the diagnostics that decide which key fix to try next.

One fp16 greedy pass per sentence, no quantization on the path. The talker's
eager attention is hooked; after each sentence the full post-RoPE K/V cache of
every talker layer is read off the last attention call, a pre-RoPE copy is made
(RoPE undone with the layer's own tables), and keygauss_core measures per
(layer, kv_head, tensor, segment):

  * Table-1 analogue over 32-token windows: retained variance, W1, KS against
    the N(0, 1/d) target of the fixed codebook (plus a true-Gaussian reference);
  * offset (energy of the per-coordinate mean) and drift (share of variance
    between windows) that separate "wrong location/scale" from "moves in time";
  * quantization error of fixed Lloyd-Max vs per-group min-max (the paper's
    fix) vs a static per-coordinate affine map calibrated on the first
    --calib-frac of the sentences (the zero-metadata alternative);
  * per layer: query effective rank, key effective rank, share of key variance
    in the top-8 query directions, and attention mass on the first 64 keys and
    on the prompt (sink / prefix protection).

Code-predictor layers (Qwen3TTSAttention) are ignored; only the talker class
is recorded. Segments: prompt = prefill tokens (text + reference audio),
gen = generated codec tokens.

Run from the repo root on the GPU box (see scripts/gpu/15_qwen_keygauss.sh):
  python models/Qwen3-TTS/benchmarks/key_gauss_stats.py --data-dir data \
      --groups librispeech_pc --max-per-group 20 --out results/keygauss_stats.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import qwen_tts.core.models.modeling_qwen3_tts as modeling  # noqa: E402
from qwen_tts.core.models.turboquant_kv_cache import rope_keys  # noqa: E402

from benchmark_qwen3tts_real import Qwen3TTSModel, _resolve_voice, run_generation  # noqa: E402
from keygauss_core import KeyStatsRecorder, make_patch, summarize  # noqa: E402
from kv_attn_divergence_experiment import force_eager, resolve_n_layers  # noqa: E402

from turboquant.bench_common import decode_overrides, set_global_seed  # noqa: E402
from turboquant.eval_sentences import iter_eval_items  # noqa: E402


# --------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--groups", default="librispeech_pc")
    ap.add_argument("--max-per-group", type=int, default=20)
    ap.add_argument("--calib-frac", type=float, default=0.5, help="share of sentences used only to calibrate the affine map")
    ap.add_argument("--group", type=int, default=32)
    ap.add_argument("--bits", default="2,3")
    ap.add_argument("--first", type=int, default=64, help="attention-mass window at the start of the cache")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rotation-seed", type=int, default=42, help="TurboQuantV3 default seed for the key rotation")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--voice-mode", default="clone")
    ap.add_argument("--default-ref", default=None)
    ap.add_argument("--talker-class", default="Qwen3TTSTalkerAttention")
    ap.add_argument("--out", default="results/keygauss_stats.csv")
    ap.add_argument("--layers-out", default=None, help="default: <out stem>_layers.csv")
    ap.add_argument("--summary", default=None, help="default: <out stem>_summary.md")
    args = ap.parse_args()

    bits_list = [int(b) for b in args.bits.split(",") if b.strip()]
    stem = os.path.splitext(args.out)[0]
    layers_out = args.layers_out or f"{stem}_layers.csv"
    summary_out = args.summary or f"{stem}_summary.md"
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    items = [(g, i, it) for g in groups for i, it in enumerate(iter_eval_items([g], args.max_per_group, args.data_dir))]
    n_calib = int(round(args.calib_frac * len(items)))
    print(f"{len(items)} sentences, first {n_calib} calibrate the affine map, rest evaluate it")

    print(f"loading {args.model} on {args.device} ({args.dtype})")
    model = Qwen3TTSModel.from_pretrained(args.model, device_map=args.device, dtype=dtype)
    speakers = model.get_supported_speakers()
    speaker = speakers[0] if speakers else "Ryan"
    print(f"talker n_layers = {resolve_n_layers(model)}")
    recorder = KeyStatsRecorder(args.talker_class, bits_list, args.group, args.first,
                                args.rotation_seed, n_calib, rope_fn=rope_keys)
    force_eager(model)
    original = modeling.eager_attention_forward
    modeling.eager_attention_forward = make_patch(original, recorder)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    all_rows, all_layers = [], []
    try:
        for group, idx, item in items:
            ref_audio, ref_text = _resolve_voice(item, args.voice_mode, args.default_ref, None)
            if args.voice_mode == "clone" and not ref_audio:
                continue
            recorder.reset()
            recorder.active = True
            set_global_seed(args.seed, deterministic=False)
            run_generation(model, item.text, "English", speaker, None, seed=args.seed,
                           gen_overrides=decode_overrides("greedy"), ref_audio=ref_audio, ref_text=ref_text)
            recorder.active = False
            rows, ls = recorder.finalize(dict(group=group, idx=idx))
            all_rows += rows
            all_layers += ls
            pd.DataFrame(all_rows).to_csv(args.out, index=False)
            pd.DataFrame(all_layers).to_csv(layers_out, index=False)
            print(f"  {group}/{idx}: {ls[0]['n_tok']} tokens ({ls[0]['prompt_len']} prompt), "
                  f"{len(ls)} layers, pre_rope_ok={ls[0]['pre_rope_ok']}")
    finally:
        modeling.eager_attention_forward = original
    print(f"attention classes seen: {recorder.seen_classes}")
    if recorder.errors:
        print(f"WARNING: {recorder.errors} measurement errors\n{recorder.first_error}")
    stats, layers = pd.DataFrame(all_rows), pd.DataFrame(all_layers)
    md = summarize(stats, layers, args.group)
    with open(summary_out, "w") as f:
        f.write(md)
    print(md)
    print(f"\nrows -> {args.out}\nlayers -> {layers_out}\nsummary -> {summary_out}")


if __name__ == "__main__":
    main()
