"""TurboQuant-compressed KV cache for Qwen3-TTS.

Drop-in replacement for DynamicCache that compresses key/value states
using TurboQuant's MSE-optimal vector quantization (Lloyd-Max + random rotation).

Typical usage:
    config = TurboQuantConfig(key_bits=4, value_bits=2)
    cache = TurboQuantKVCache(config, n_layers=20)
    # pass as past_key_values to model.generate()
"""

from typing import Optional

import torch
from transformers.cache_utils import DynamicCache

from turboquant.config import TurboQuantConfig
from turboquant.compressors_v3 import TurboQuantV3


class TurboQuantKVCache(DynamicCache):
    """KV cache that compresses older tokens via TurboQuant vector quantization.

    Keeps the most recent `residual_window` tokens in fp16 for generation quality.
    Tokens that fall outside the window are compressed using MSE-optimal quantization
    (random rotation + Lloyd-Max scalar quantization per coordinate).

    This is a drop-in replacement for DynamicCache -- the attention mechanism
    sees full-precision K/V tensors reconstructed from compressed + fp16 parts.
    """

    def __init__(self, config: TurboQuantConfig, n_layers: int = 20):
        super().__init__()
        self.config = config
        self.n_layers = n_layers

        # Per-layer storage
        self._compressors: dict[int, TurboQuantV3] = {}
        self._chunks_k: dict[int, list[dict]] = {}
        self._chunks_v: dict[int, list[dict]] = {}
        self._fp16_recent_k: dict[int, list[torch.Tensor]] = {}
        self._fp16_recent_v: dict[int, list[torch.Tensor]] = {}
        self._total_seq: dict[int, int] = {}
        # Cached decompressed prefix — avoids re-decompressing every step
        self._decompressed_k: dict[int, Optional[torch.Tensor]] = {}
        self._decompressed_v: dict[int, Optional[torch.Tensor]] = {}

    def _get_compressor(self, layer_idx: int, head_dim: int, device: torch.device) -> TurboQuantV3:
        """Lazy-initialize a TurboQuantV3 compressor for the given layer."""
        if layer_idx not in self._compressors:
            self._compressors[layer_idx] = TurboQuantV3(
                head_dim=head_dim,
                key_bits=self.config.key_bits,
                value_bits=self.config.value_bits,
                residual_window=0,  # we manage windowing ourselves
                layer_idx=layer_idx,
                n_layers=self.n_layers,
                protected_layers=self.config.protected_layers,
                protected_bits=self.config.protected_bits,
                seed=self.config.seed,
                device=str(device),
            )
        return self._compressors[layer_idx]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new K/V states, compress overflow, return full reconstructed K/V.

        Args:
            key_states: (batch, num_kv_heads, seq_new, head_dim)
            value_states: (batch, num_kv_heads, seq_new, head_dim)
            layer_idx: transformer layer index
            cache_kwargs: unused, kept for API compatibility

        Returns:
            (full_keys, full_values) with shape (batch, num_kv_heads, total_seq, head_dim)
        """
        _, _, s_new, head_dim = key_states.shape
        comp = self._get_compressor(layer_idx, head_dim, key_states.device)

        # Initialize per-layer storage on first call
        if layer_idx not in self._chunks_k:
            self._chunks_k[layer_idx] = []
            self._chunks_v[layer_idx] = []
            self._fp16_recent_k[layer_idx] = []
            self._fp16_recent_v[layer_idx] = []
            self._total_seq[layer_idx] = 0
            self._decompressed_k[layer_idx] = None
            self._decompressed_v[layer_idx] = None

        self._total_seq[layer_idx] += s_new

        # Append to fp16 recent buffer
        self._fp16_recent_k[layer_idx].append(key_states)
        self._fp16_recent_v[layer_idx].append(value_states)

        # Concatenate recent buffer
        recent_k = torch.cat(self._fp16_recent_k[layer_idx], dim=2)
        recent_v = torch.cat(self._fp16_recent_v[layer_idx], dim=2)
        rw = self.config.residual_window

        # Compress overflow beyond residual window as a single chunk
        compressed_this_step = False
        if recent_k.shape[2] > rw:
            overflow = recent_k.shape[2] - rw
            overflow_k = recent_k[:, :, :overflow, :]
            overflow_v = recent_v[:, :, :overflow, :]

            ck, cv = comp.compress_kv(overflow_k, overflow_v)
            self._chunks_k[layer_idx].append(ck)
            self._chunks_v[layer_idx].append(cv)

            # Keep only the residual window in fp16
            recent_k = recent_k[:, :, overflow:, :]
            recent_v = recent_v[:, :, overflow:, :]
            self._fp16_recent_k[layer_idx] = [recent_k]
            self._fp16_recent_v[layer_idx] = [recent_v]
            compressed_this_step = True

            # Decompress only the new chunk and append to cached prefix (on CPU)
            dk, dv = comp.decompress_kv(ck, cv)
            dk = dk.to(key_states.dtype).cpu()
            dv = dv.to(value_states.dtype).cpu()

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

        # Build full K/V: move decompressed prefix to GPU + fp16 recent window
        recent_k = torch.cat(self._fp16_recent_k[layer_idx], dim=2)
        recent_v = torch.cat(self._fp16_recent_v[layer_idx], dim=2)
        device = recent_k.device

        if self._decompressed_k[layer_idx] is not None:
            prefix_k = self._decompressed_k[layer_idx].to(device)
            prefix_v = self._decompressed_v[layer_idx].to(device)
            full_k = torch.cat([prefix_k, recent_k], dim=2)
            full_v = torch.cat([prefix_v, recent_v], dim=2)
            del prefix_k, prefix_v
        else:
            full_k = recent_k
            full_v = recent_v

        # Extend DynamicCache's internal layer list for transformers bookkeeping
        self._ensure_layer_stubs(layer_idx)

        return full_k, full_v

    def _ensure_layer_stubs(self, layer_idx: int) -> None:
        """Ensure self.layers has enough entries for transformers internals."""
        from transformers.cache_utils import DynamicLayer
        while len(self.layers) <= layer_idx:
            self.layers.append(DynamicLayer())

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._total_seq.get(layer_idx, 0)

    @property
    def evicted_count(self) -> int:
        """Always 0 -- TurboQuant compresses tokens, it doesn't evict them.

        The full sequence is always reconstructed before attention, so
        RoPE positions remain correct without any correction.
        """
        return 0

    @staticmethod
    def _count_chunk_bytes(chunk: dict) -> tuple[int, int]:
        """Count (compressed_bytes, fp16_bytes) from a TurboQuantV3 chunk dict.

        TurboQuantV3.compress_kv returns dicts like:
            {"compressed": {"idx_bytes": Tensor, "vec_norms": Tensor, ...},
             "fp16": Tensor, "shape": ..., "split_at": int}
        """
        compressed_bytes = 0
        fp16_bytes = 0

        inner = chunk.get("compressed")
        if isinstance(inner, dict):
            for v in inner.values():
                if isinstance(v, torch.Tensor):
                    compressed_bytes += v.nelement() * v.element_size()

        fp16_tensor = chunk.get("fp16")
        if isinstance(fp16_tensor, torch.Tensor):
            fp16_bytes += fp16_tensor.nelement() * fp16_tensor.element_size()

        return compressed_bytes, fp16_bytes

    def memory_report(self) -> dict:
        """Report compressed vs fp16 memory usage across all layers."""
        total_compressed_bytes = 0
        total_fp16_bytes = 0

        for layer_idx in self._total_seq:
            # fp16 recent window (uncompressed)
            for t in self._fp16_recent_k.get(layer_idx, []):
                total_fp16_bytes += t.nelement() * t.element_size()
            for t in self._fp16_recent_v.get(layer_idx, []):
                total_fp16_bytes += t.nelement() * t.element_size()

            # Compressed chunks
            for chunk in self._chunks_k.get(layer_idx, []):
                cb, fb = self._count_chunk_bytes(chunk)
                total_compressed_bytes += cb
                total_fp16_bytes += fb
            for chunk in self._chunks_v.get(layer_idx, []):
                cb, fb = self._count_chunk_bytes(chunk)
                total_compressed_bytes += cb
                total_fp16_bytes += fb

        total_bytes = total_compressed_bytes + total_fp16_bytes
        fp16_equivalent = sum(
            seq * 2 * 128 * 2  # seq_len * num_kv_heads * head_dim * 2 bytes (fp16)
            for seq in self._total_seq.values()
        ) * 2  # keys + values

        return {
            "compressed_bytes": total_compressed_bytes,
            "fp16_recent_bytes": total_fp16_bytes,
            "total_bytes": total_bytes,
            "fp16_equivalent_bytes": fp16_equivalent,
            "compression_ratio": fp16_equivalent / total_bytes if total_bytes > 0 else 1.0,
        }
