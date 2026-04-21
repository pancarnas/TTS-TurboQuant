"""TurboQuant KV cache compression for VALL-E-X.

VALL-E-X uses a tuple-based KV cache: tuple of 12 (key, value) pairs,
where each key/value has shape (B, num_heads=16, seq_len, head_dim=64).

This module provides a cache manager that stores compressed chunks internally
and provides reconstructed K/V on demand, avoiding full-tensor duplication.
"""

import sys
import os
from typing import Optional

import torch

# Add project root to find shared turboquant
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from turboquant.config import TurboQuantConfig
from turboquant.compressors_v3 import TurboQuantV3


class TurboQuantValleCache:
    """Manages TurboQuant compression for VALL-E-X's KV cache.

    Instead of compressing after the attention layer builds a full tensor,
    this cache manages storage incrementally:
    - Accepts new K/V tokens via update()
    - Compresses overflow beyond the residual window
    - Returns full reconstructed K/V for attention
    - Caches the decompressed prefix to avoid re-decompressing every step
    """

    def __init__(self, config: TurboQuantConfig, n_layers: int = 12):
        self.config = config
        self.n_layers = n_layers
        self._compressors: dict[int, TurboQuantV3] = {}
        self._chunks_k: dict[int, list[dict]] = {}
        self._chunks_v: dict[int, list[dict]] = {}
        self._fp16_recent_k: dict[int, list[torch.Tensor]] = {}
        self._fp16_recent_v: dict[int, list[torch.Tensor]] = {}
        self._decompressed_k: dict[int, Optional[torch.Tensor]] = {}
        self._decompressed_v: dict[int, Optional[torch.Tensor]] = {}
        self._total_seq: dict[int, int] = {}

    def _get_compressor(self, layer_idx: int, head_dim: int, device: torch.device) -> TurboQuantV3:
        if layer_idx not in self._compressors:
            self._compressors[layer_idx] = TurboQuantV3(
                head_dim=head_dim,
                key_bits=self.config.key_bits,
                value_bits=self.config.value_bits,
                residual_window=0,
                layer_idx=layer_idx,
                n_layers=self.n_layers,
                protected_layers=self.config.protected_layers,
                protected_bits=self.config.protected_bits,
                seed=self.config.seed,
                device=str(device),
            )
        return self._compressors[layer_idx]

    def update(self, layer_idx: int, new_key: torch.Tensor, new_value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new K/V tokens, compress overflow, return full reconstructed K/V.

        Args:
            layer_idx: transformer layer index
            new_key: (B, num_heads, seq_new, head_dim) — only the NEW tokens
            new_value: (B, num_heads, seq_new, head_dim)

        Returns:
            (full_key, full_value) for attention computation
        """
        head_dim = new_key.shape[-1]
        comp = self._get_compressor(layer_idx, head_dim, new_key.device)
        rw = self.config.residual_window

        if layer_idx not in self._chunks_k:
            self._chunks_k[layer_idx] = []
            self._chunks_v[layer_idx] = []
            self._fp16_recent_k[layer_idx] = []
            self._fp16_recent_v[layer_idx] = []
            self._decompressed_k[layer_idx] = None
            self._decompressed_v[layer_idx] = None
            self._total_seq[layer_idx] = 0

        self._total_seq[layer_idx] += new_key.shape[2]

        # Append to fp16 recent buffer
        self._fp16_recent_k[layer_idx].append(new_key)
        self._fp16_recent_v[layer_idx].append(new_value)

        # Concatenate recent buffer
        recent_k = torch.cat(self._fp16_recent_k[layer_idx], dim=2)
        recent_v = torch.cat(self._fp16_recent_v[layer_idx], dim=2)

        # Compress overflow beyond residual window
        if recent_k.shape[2] > rw:
            overflow = recent_k.shape[2] - rw
            overflow_k = recent_k[:, :, :overflow, :]
            overflow_v = recent_v[:, :, :overflow, :]

            ck, cv = comp.compress_kv(overflow_k, overflow_v)
            self._chunks_k[layer_idx].append(ck)
            self._chunks_v[layer_idx].append(cv)

            # Keep only residual window in fp16
            recent_k = recent_k[:, :, overflow:, :]
            recent_v = recent_v[:, :, overflow:, :]
            self._fp16_recent_k[layer_idx] = [recent_k]
            self._fp16_recent_v[layer_idx] = [recent_v]

            # Decompress only the new chunk and append to cached prefix
            dk, dv = comp.decompress_kv(ck, cv)
            dk = dk.to(new_key.dtype)
            dv = dv.to(new_value.dtype)

            if self._decompressed_k[layer_idx] is not None:
                self._decompressed_k[layer_idx] = torch.cat(
                    [self._decompressed_k[layer_idx], dk], dim=2
                )
                self._decompressed_v[layer_idx] = torch.cat(
                    [self._decompressed_v[layer_idx], dv], dim=2
                )
            else:
                self._decompressed_k[layer_idx] = dk
                self._decompressed_v[layer_idx] = dv

        # Build full K/V: cached decompressed prefix + fp16 recent
        recent_k = torch.cat(self._fp16_recent_k[layer_idx], dim=2)
        recent_v = torch.cat(self._fp16_recent_v[layer_idx], dim=2)

        if self._decompressed_k[layer_idx] is not None:
            full_k = torch.cat([self._decompressed_k[layer_idx], recent_k], dim=2)
            full_v = torch.cat([self._decompressed_v[layer_idx], recent_v], dim=2)
        else:
            full_k = recent_k
            full_v = recent_v

        return full_k, full_v

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._total_seq.get(layer_idx, 0)

    def memory_report(self) -> dict:
        """Report compression statistics."""
        total_compressed_bytes = 0
        total_fp16_bytes = 0

        for layer_idx in self._total_seq:
            for t in self._fp16_recent_k.get(layer_idx, []):
                total_fp16_bytes += t.nelement() * t.element_size()
            for t in self._fp16_recent_v.get(layer_idx, []):
                total_fp16_bytes += t.nelement() * t.element_size()

            for chunk in self._chunks_k.get(layer_idx, []):
                inner = chunk.get("compressed")
                if isinstance(inner, dict):
                    for v in inner.values():
                        if isinstance(v, torch.Tensor):
                            total_compressed_bytes += v.nelement() * v.element_size()
                fp16_t = chunk.get("fp16")
                if isinstance(fp16_t, torch.Tensor):
                    total_fp16_bytes += fp16_t.nelement() * fp16_t.element_size()
            for chunk in self._chunks_v.get(layer_idx, []):
                inner = chunk.get("compressed")
                if isinstance(inner, dict):
                    for v in inner.values():
                        if isinstance(v, torch.Tensor):
                            total_compressed_bytes += v.nelement() * v.element_size()
                fp16_t = chunk.get("fp16")
                if isinstance(fp16_t, torch.Tensor):
                    total_fp16_bytes += fp16_t.nelement() * fp16_t.element_size()

        total_bytes = total_compressed_bytes + total_fp16_bytes
        fp16_equivalent = sum(
            seq * 16 * 64 * 2  # seq_len * num_heads * head_dim * 2 bytes (fp16)
            for seq in self._total_seq.values()
        ) * 2  # keys + values

        return {
            "compressed_bytes": total_compressed_bytes,
            "fp16_recent_bytes": total_fp16_bytes,
            "total_bytes": total_bytes,
            "fp16_equivalent_bytes": fp16_equivalent,
            "compression_ratio": fp16_equivalent / total_bytes if total_bytes > 0 else 1.0,
        }
