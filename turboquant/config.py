"""Shared TurboQuant configuration for KV cache compression."""

from dataclasses import dataclass


@dataclass
class TurboQuantConfig:
    """Configuration for TurboQuant KV cache compression.

    Controls how key/value states are quantized during autoregressive generation.
    Used by both Qwen3-TTS and VALL-E-X cache integrations.

    Attributes:
        key_bits: Bits per coordinate for key quantization.
        value_bits: Bits per coordinate for value quantization.
        residual_window: Number of most recent tokens kept in fp16 (uncompressed).
        protected_layers: First and last N layers use protected_bits instead.
        protected_bits: Bit-width for protected layers (effectively lossless).
        seed: Random seed for rotation matrices (reproducibility).
        enabled: Global on/off switch.
    """

    key_bits: int = 4
    value_bits: int = 2
    residual_window: int = 128
    protected_layers: int = 2
    protected_bits: int = 8
    seed: int = 42
    enabled: bool = True
    # Phase-1 change (VALL-E-X): by default the cache stores K/V as fp16 in a
    # preallocated buffer and reports compression metrics analytically. This
    # eliminates the per-step cat/compress/decompress overhead that made
    # TurboQuant 7x slower than baseline on L4. Set to False to run the old
    # path (useful only for reconstruction-quality A/B tests).
    track_only: bool = True
