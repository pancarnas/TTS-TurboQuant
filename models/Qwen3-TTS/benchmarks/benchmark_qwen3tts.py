"""Benchmark TurboQuant KV cache compression on Qwen3-TTS.

Measures compression ratio, reconstruction quality, and latency
for different bit-width configurations.

Usage:
    python benchmarks/benchmark_qwen3tts.py [--device cuda]
"""

import time
import argparse

import torch
from qwen_tts.core.models.turboquant_kv_cache import TurboQuantConfig, TurboQuantKVCache

# Qwen3-TTS architecture constants
BATCH = 1
NUM_KV_HEADS = 2
HEAD_DIM = 128
N_LAYERS = 20


def fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    elif n < 1024 ** 2:
        return f"{n / 1024:.1f} KB"
    else:
        return f"{n / 1024 ** 2:.1f} MB"


def benchmark_compression(device: str):
    """Measure compression ratio with bulk prefill."""
    configs = [
        ("K4/V2 (default)", TurboQuantConfig(key_bits=4, value_bits=2, residual_window=128)),
        ("K4/V2 rw=64",     TurboQuantConfig(key_bits=4, value_bits=2, residual_window=64)),
        ("K3/V2",           TurboQuantConfig(key_bits=3, value_bits=2, residual_window=128)),
        ("K4/V4",           TurboQuantConfig(key_bits=4, value_bits=4, residual_window=128)),
        ("K2/V2",           TurboQuantConfig(key_bits=2, value_bits=2, residual_window=128)),
    ]
    seq_lengths = [500, 1000, 2000, 4000]

    print("=" * 90)
    print(f"Qwen3-TTS KV Cache Compression Benchmark (device={device})")
    print(f"Architecture: {N_LAYERS} layers, {NUM_KV_HEADS} KV heads, head_dim={HEAD_DIM}")
    print("=" * 90)

    for total in seq_lengths:
        print(f"\n--- {total} tokens ---")
        print(f"{'Config':<20} {'Compressed':<12} {'FP16 Recent':<12} {'Total':<12} {'FP16 Equiv':<12} {'Ratio':<8} {'Time':<8}")
        print("-" * 90)

        for name, cfg in configs:
            cache = TurboQuantKVCache(cfg, n_layers=N_LAYERS)
            k = torch.randn(BATCH, NUM_KV_HEADS, total, HEAD_DIM, device=device)
            v = torch.randn(BATCH, NUM_KV_HEADS, total, HEAD_DIM, device=device)

            t0 = time.time()
            for layer_idx in range(N_LAYERS):
                cache.update(k, v, layer_idx)
            elapsed = time.time() - t0

            report = cache.memory_report()
            print(
                f"{name:<20} "
                f"{fmt_bytes(report['compressed_bytes']):<12} "
                f"{fmt_bytes(report['fp16_recent_bytes']):<12} "
                f"{fmt_bytes(report['total_bytes']):<12} "
                f"{fmt_bytes(report['fp16_equivalent_bytes']):<12} "
                f"{report['compression_ratio']:<8.2f}x"
                f"{elapsed:<8.2f}s"
            )


def benchmark_latency(device: str):
    """Measure per-step decode latency with TurboQuant."""
    cfg = TurboQuantConfig(key_bits=4, value_bits=2, residual_window=128)
    prefill_len = 100

    print(f"\n{'=' * 60}")
    print(f"Decode Latency Benchmark (K4/V2, rw=128, device={device})")
    print(f"{'=' * 60}")
    print(f"{'Decode Steps':<15} {'Total ms':<12} {'ms/step':<12}")
    print("-" * 40)

    for decode_steps in [100, 500, 1000]:
        cache = TurboQuantKVCache(cfg, n_layers=N_LAYERS)

        # Prefill
        for layer_idx in range(N_LAYERS):
            k = torch.randn(BATCH, NUM_KV_HEADS, prefill_len, HEAD_DIM, device=device)
            v = torch.randn(BATCH, NUM_KV_HEADS, prefill_len, HEAD_DIM, device=device)
            cache.update(k, v, layer_idx)

        # Decode
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for step in range(decode_steps):
            for layer_idx in range(N_LAYERS):
                k = torch.randn(BATCH, NUM_KV_HEADS, 1, HEAD_DIM, device=device)
                v = torch.randn(BATCH, NUM_KV_HEADS, 1, HEAD_DIM, device=device)
                cache.update(k, v, layer_idx)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.time() - t0) * 1000

        print(f"{decode_steps:<15} {elapsed:<12.1f} {elapsed / decode_steps:<12.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    benchmark_compression(args.device)
    benchmark_latency(args.device)


if __name__ == "__main__":
    main()
