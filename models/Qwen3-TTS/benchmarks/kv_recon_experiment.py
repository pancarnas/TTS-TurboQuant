"""Focused KV-reconstruction experiment: K4/V4 at residual_window=24 vs 0.

For each sentence it runs ONE uncompressed (fp16) generation to build the real KV
cache, then compresses+decompresses that SAME cache under K4/V4 with each residual
window, measuring per-layer cosine similarity + relative MSE. This is the
deterministic, sampling-free rate-distortion measurement: it shows how much the
residual window reduces *static* reconstruction error.

It needs no baseline trial (the fp16 pass is just the substrate) and no Whisper/
WavLM. IMPORTANT: a small static rw0-vs-rw24 reconstruction gap does NOT explain
the rw=0 generation collapse — that comes from autoregressive error amplification,
which this does not measure. Pair this with the generation collapse rates and
report the gap between them.

Run from the repo root (same env as the benchmark):
  python models/Qwen3-TTS/benchmarks/kv_recon_experiment.py \
      --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --data-dir data \
      --groups seedtts_en,librispeech_pc,libritts_long,ellav_hard \
      --max-per-group 100 --residual-windows 24,0 --out results/kv_recon_k4v4.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark_qwen3tts_real import (  # noqa: E402 - needs sys.path tweak above
    Qwen3TTSModel,
    _extract_fp16_kv,
    _kv_recon_errors,
    _resolve_voice,
    run_generation,
)

from turboquant.bench_common import decode_overrides, set_global_seed  # noqa: E402
from turboquant.compressors_v3 import TurboQuantV3  # noqa: E402
from turboquant.eval_sentences import iter_eval_items  # noqa: E402

COLUMNS = [
    "group",
    "idx",
    "seed",
    "key_bits",
    "value_bits",
    "residual_window",
    "layer",
    "cos_k",
    "cos_v",
    "relmse_k",
    "relmse_v",
]


def build_fp16_cache(model, item, speaker, seed, voice_mode, default_ref):
    """Run one greedy fp16 generation to populate the cache; return (n, keys, vals)."""
    ref_audio, ref_text = _resolve_voice(item, voice_mode, default_ref, None)
    if voice_mode == "clone" and not ref_audio:
        return 0, [], []
    set_global_seed(seed, deterministic=False)
    run_generation(
        model,
        item.text,
        "English",
        speaker,
        None,
        seed=seed,
        gen_overrides=decode_overrides("greedy"),
        ref_audio=ref_audio,
        ref_text=ref_text,
    )
    return _extract_fp16_kv(getattr(model.model, "last_kv_cache", None))


def recon_for_rw(keys, vals, kb, vb, rw, n_layers, head_dim, prot_layers, prot_bits):
    """Compress+decompress every layer at one residual window; per-layer error rows."""
    rows = []
    for layer in range(n_layers):
        comp = TurboQuantV3(
            head_dim=head_dim,
            key_bits=kb,
            value_bits=vb,
            residual_window=rw,
            layer_idx=layer,
            n_layers=n_layers,
            protected_layers=prot_layers,
            protected_bits=prot_bits,
            seed=42,
            device=str(keys[layer].device),
        )
        ck, cv = comp.compress_kv(keys[layer], vals[layer])
        rk, rv = comp.decompress_kv(ck, cv)
        cos_k, relmse_k = _kv_recon_errors(keys[layer], rk, head_dim)
        cos_v, relmse_v = _kv_recon_errors(vals[layer], rv, head_dim)
        rows.append((layer, cos_k, cos_v, relmse_k, relmse_v))
    return rows


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per-residual-window mean error + a 'diff (rw0 − rwN)' row when both present."""
    agg = (
        df.groupby("residual_window")[["cos_k", "cos_v", "relmse_k", "relmse_v"]]
        .mean()
        .sort_index(ascending=False)
    )
    if 0 in agg.index and len(agg.index) > 1:
        other = max(w for w in agg.index if w != 0)
        agg.loc[f"diff(0-{other})"] = agg.loc[0] - agg.loc[other]
    return agg


def run_experiment(model, args, speaker) -> pd.DataFrame:
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    rws = [int(w) for w in args.residual_windows.split(",") if w.strip()]
    items = [
        (g, i, it)
        for g in groups
        for i, it in enumerate(iter_eval_items([g], args.max_per_group, args.data_dir))
    ]
    out_rows, missing = [], 0
    it = tqdm(items, desc="kv-recon", unit="sent") if tqdm else items
    for group, idx, item in it:
        n_layers, keys, vals = build_fp16_cache(
            model, item, speaker, args.seed, args.voice_mode, args.default_ref
        )
        if n_layers == 0:
            missing += 1
            continue
        head_dim = keys[0].shape[-1]
        for rw in rws:
            for layer, ck, cv, rk, rv in recon_for_rw(
                keys,
                vals,
                args.key_bits,
                args.value_bits,
                rw,
                n_layers,
                head_dim,
                args.protected_layers,
                args.protected_bits,
            ):
                out_rows.append(
                    (
                        group,
                        idx,
                        args.seed,
                        args.key_bits,
                        args.value_bits,
                        rw,
                        layer,
                        ck,
                        cv,
                        rk,
                        rv,
                    )
                )
    if missing:
        print(f"  WARNING: {missing} sentences had no extractable cache")
    return pd.DataFrame(out_rows, columns=COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--groups", default="seedtts_en,librispeech_pc,libritts_long,ellav_hard"
    )
    parser.add_argument("--max-per-group", type=int, default=100)
    parser.add_argument("--residual-windows", default="24,0")
    parser.add_argument("--key-bits", type=int, default=4)
    parser.add_argument("--value-bits", type=int, default=4)
    parser.add_argument("--protected-layers", type=int, default=2)
    parser.add_argument("--protected-bits", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--voice-mode", default="clone")
    parser.add_argument("--default-ref", default=None)
    parser.add_argument("--out", default="results/kv_recon_k4v4.csv")
    args = parser.parse_args()

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(args.dtype, torch.bfloat16)
    print(f"loading {args.model} on {args.device} ({args.dtype})")
    model = Qwen3TTSModel.from_pretrained(
        args.model, device_map=args.device, dtype=dtype
    )
    speakers = model.get_supported_speakers()
    speaker = speakers[0] if speakers else "Ryan"

    df = run_experiment(model, args, speaker)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {len(df)} per-layer rows -> {args.out}")
    print(
        f"\n== mean reconstruction by residual window (K{args.key_bits}/V{args.value_bits}) =="
    )
    print(summarize(df).to_string(float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
