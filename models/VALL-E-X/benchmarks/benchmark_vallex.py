"""Benchmark TurboQuant KV cache compression on VALL-E-X.

Measures compression ratio, reconstruction quality, and latency
for different bit-width configurations.

Usage:
    python models/VALL-E-X/benchmarks/benchmark_vallex.py [--device cuda]
"""

import sys
import os
import time
import argparse

# VALL-E-X uses bare module imports; add its directory to sys.path
_VALLEX_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _VALLEX_DIR)

import torch
from turboquant_cache import TurboQuantConfig, TurboQuantValleCache

# VALL-E-X architecture constants
BATCH = 1
NUM_HEADS = 16
HEAD_DIM = 64
N_LAYERS = 12


def fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    elif n < 1024 ** 2:
        return f"{n / 1024:.1f} KB"
    else:
        return f"{n / 1024 ** 2:.1f} MB"


def benchmark_compression(device: str):
    """Measure compression ratio with simulated prefill."""
    configs = [
        ("K4/V2 (default)", TurboQuantConfig(key_bits=4, value_bits=2, residual_window=128)),
        ("K4/V2 rw=64",     TurboQuantConfig(key_bits=4, value_bits=2, residual_window=64)),
        ("K3/V2",           TurboQuantConfig(key_bits=3, value_bits=2, residual_window=128)),
        ("K4/V4",           TurboQuantConfig(key_bits=4, value_bits=4, residual_window=128)),
        ("K2/V2",           TurboQuantConfig(key_bits=2, value_bits=2, residual_window=128)),
    ]
    seq_lengths = [500, 1000, 2000, 4000]

    print("=" * 90)
    print(f"VALL-E-X KV Cache Compression Benchmark (device={device})")
    print(f"Architecture: {N_LAYERS} layers, {NUM_HEADS} heads, head_dim={HEAD_DIM}")
    print("=" * 90)

    for total in seq_lengths:
        print(f"\n--- {total} tokens ---")
        print(f"{'Config':<20} {'Compressed':<12} {'FP16 Recent':<12} {'Total':<12} {'FP16 Equiv':<12} {'Ratio':<8} {'Time':<8}")
        print("-" * 90)

        for name, cfg in configs:
            manager = TurboQuantValleCache(cfg, n_layers=N_LAYERS)

            # Simulate prefill: feed all tokens at once per layer
            k = torch.randn(BATCH, NUM_HEADS, total, HEAD_DIM, device=device)
            v = torch.randn(BATCH, NUM_HEADS, total, HEAD_DIM, device=device)

            t0 = time.time()
            for layer_idx in range(N_LAYERS):
                manager.update(layer_idx, k, v)
            elapsed = time.time() - t0

            report = manager.memory_report()
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
    """Simulate autoregressive decode latency with compression."""
    cfg = TurboQuantConfig(key_bits=4, value_bits=2, residual_window=128)
    prefill_len = 100

    print(f"\n{'=' * 60}")
    print(f"Decode Latency Benchmark (K4/V2, rw=128, device={device})")
    print(f"{'=' * 60}")
    print(f"{'Decode Steps':<15} {'Total ms':<12} {'ms/step':<12}")
    print("-" * 40)

    for decode_steps in [100, 500, 1000]:
        manager = TurboQuantValleCache(cfg, n_layers=N_LAYERS)

        # Prefill
        k = torch.randn(BATCH, NUM_HEADS, prefill_len, HEAD_DIM, device=device)
        v = torch.randn(BATCH, NUM_HEADS, prefill_len, HEAD_DIM, device=device)
        for layer_idx in range(N_LAYERS):
            manager.update(layer_idx, k, v)

        # Decode
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for step in range(decode_steps):
            for layer_idx in range(N_LAYERS):
                new_k = torch.randn(BATCH, NUM_HEADS, 1, HEAD_DIM, device=device)
                new_v = torch.randn(BATCH, NUM_HEADS, 1, HEAD_DIM, device=device)
                manager.update(layer_idx, new_k, new_v)
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
