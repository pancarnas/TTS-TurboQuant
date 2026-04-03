"""Benchmark TurboQuant KV cache compression efficiency.

Uses bulk prefill (no per-token decode loop) to measure memory savings fast.
Single layer is sufficient since compression ratio is layer-independent;
the report is then scaled to 20 layers.
"""

import time
import torch
from qwen_tts.core.models.turboquant_kv_cache import TurboQuantConfig, TurboQuantKVCache

# Qwen3-TTS architecture constants
BATCH = 1
NUM_KV_HEADS = 2
HEAD_DIM = 128
N_LAYERS = 20


def measure_compression(config: TurboQuantConfig, total_tokens: int) -> dict:
    """Feed all tokens at once via prefill (fast) and return memory report."""
    cache = TurboQuantKVCache(config, n_layers=N_LAYERS)

    # Single bulk insert per layer -- same compression result as token-by-token
    k = torch.randn(BATCH, NUM_KV_HEADS, total_tokens, HEAD_DIM)
    v = torch.randn(BATCH, NUM_KV_HEADS, total_tokens, HEAD_DIM)

    for layer_idx in range(N_LAYERS):
        cache.update(k, v, layer_idx)

    return cache.memory_report()


def fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    elif n < 1024 ** 2:
        return f"{n / 1024:.1f} KB"
    else:
        return f"{n / 1024 ** 2:.1f} MB"


def main():
    configs = [
        ("K4/V2 (default)", TurboQuantConfig(key_bits=4, value_bits=2, residual_window=128)),
        ("K4/V2 rw=64",     TurboQuantConfig(key_bits=4, value_bits=2, residual_window=64)),
        ("K3/V2",           TurboQuantConfig(key_bits=3, value_bits=2, residual_window=128)),
        ("K4/V4",           TurboQuantConfig(key_bits=4, value_bits=4, residual_window=128)),
        ("K2/V2",           TurboQuantConfig(key_bits=2, value_bits=2, residual_window=128)),
    ]

    seq_lengths = [500, 1000, 2000, 4000]

    print("=" * 85)
    print("TurboQuant KV Cache Compression Benchmark")
    print(f"Model: {N_LAYERS} layers, {NUM_KV_HEADS} KV heads, head_dim={HEAD_DIM}")
    print("=" * 85)

    for total in seq_lengths:
        print(f"\n--- {total} tokens ---")
        print(f"{'Config':<20} {'Compressed':<14} {'FP16 Recent':<14} {'Total':<14} {'FP16 Equiv':<14} {'Ratio':<8}")
        print("-" * 85)

        for name, cfg in configs:
            t0 = time.time()
            report = measure_compression(cfg, total)
            elapsed = time.time() - t0
            print(
                f"{name:<20} "
                f"{fmt_bytes(report['compressed_bytes']):<14} "
                f"{fmt_bytes(report['fp16_recent_bytes']):<14} "
                f"{fmt_bytes(report['total_bytes']):<14} "
                f"{fmt_bytes(report['fp16_equivalent_bytes']):<14} "
                f"{report['compression_ratio']:<8.2f}x"
                f"  ({elapsed:.1f}s)"
            )


if __name__ == "__main__":
    main()
