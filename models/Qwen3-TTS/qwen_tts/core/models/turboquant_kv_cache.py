"""TurboQuant-compressed KV cache for Qwen3-TTS.

Drop-in replacement for HuggingFace's DynamicCache that tracks TurboQuant's
theoretical compression metrics while storing K/V in preallocated fp16 buffers.

Phase 1 rewrite (2026-04-22, see feature/valle EXPERIMENT_LOG.md): replaced the
list-of-tensors + torch.cat-per-step pattern (plus per-step CPU<->GPU round
trip on the decompressed prefix) with grow-on-doubling preallocated buffers,
one per layer, slice-written in place. Compression is off-path by default via
config.track_only=True; legacy on-path compression is still available for
reconstruction-quality A/B tests via track_only=False.

Typical usage:
    config = TurboQuantConfig(key_bits=4, value_bits=2)
    cache = TurboQuantKVCache(config, n_layers=20)
    # pass as past_key_values to model.generate()
"""

import math
from typing import Optional

import torch
from transformers.cache_utils import DynamicCache

from turboquant.config import TurboQuantConfig
from turboquant.compressors_v3 import TurboQuantV3


# Initial buffer capacity (tokens). Grows by doubling on overflow.
_INITIAL_CAPACITY = 256


class TurboQuantKVCache(DynamicCache):
    """KV cache that tracks TurboQuant compression while storing fp16 in place.

    Drop-in replacement for DynamicCache. `update()` writes new K/V into a
    preallocated per-layer buffer and returns a contiguous slice view. When
    config.track_only=True (default) the attention layer reads fp16 K/V — no
    actual compression happens per step, but memory_report() reports the
    theoretical compression ratio that TurboQuant WOULD deliver with a
    compression-aware attention kernel. When track_only=False, the legacy
    compress/decompress path runs on-path for reconstruction-quality tests.
    """

    def __init__(self, config: TurboQuantConfig, n_layers: int = 20):
        super().__init__()
        self.config = config
        self.n_layers = n_layers

        # Preallocated per-layer fp16 buffers.
        self._buf_k: dict[int, torch.Tensor] = {}
        self._buf_v: dict[int, torch.Tensor] = {}
        self._cur_len: dict[int, int] = {}

        # Virtual chunks: one entry per overflow event. In track_only=True mode
        # the entries are lightweight placeholders ({"span": int}). In
        # track_only=False mode they additionally hold the compressed blob from
        # TurboQuantV3.compress_kv for reconstruction-quality A/B tests.
        self._chunks_k: dict[int, list[dict]] = {}
        self._chunks_v: dict[int, list[dict]] = {}

        # Lazy per-layer compressor (only instantiated when track_only=False).
        self._compressors: dict[int, TurboQuantV3] = {}

    # ---------------------------------------------------------------
    # Buffer management
    # ---------------------------------------------------------------

    def _ensure_capacity(
        self, layer_idx: int, needed_len: int, template: torch.Tensor
    ) -> None:
        """Allocate or grow the layer's buffer so it holds `needed_len` tokens."""
        B, H, _, D = template.shape
        dtype = template.dtype
        device = template.device

        if layer_idx not in self._buf_k:
            cap = max(_INITIAL_CAPACITY, needed_len)
            self._buf_k[layer_idx] = torch.empty(
                B, H, cap, D, dtype=dtype, device=device
            )
            self._buf_v[layer_idx] = torch.empty(
                B, H, cap, D, dtype=dtype, device=device
            )
            self._cur_len[layer_idx] = 0
            self._chunks_k[layer_idx] = []
            self._chunks_v[layer_idx] = []
            return

        cur_cap = self._buf_k[layer_idx].shape[2]
        if needed_len <= cur_cap:
            return

        # Grow by doubling. Amortized O(1) append, O(log n) total copies.
        new_cap = cur_cap
        while new_cap < needed_len:
            new_cap *= 2
        new_k = torch.empty(B, H, new_cap, D, dtype=dtype, device=device)
        new_v = torch.empty(B, H, new_cap, D, dtype=dtype, device=device)
        s = self._cur_len[layer_idx]
        new_k[:, :, :s, :].copy_(self._buf_k[layer_idx][:, :, :s, :])
        new_v[:, :, :s, :].copy_(self._buf_v[layer_idx][:, :, :s, :])
        self._buf_k[layer_idx] = new_k
        self._buf_v[layer_idx] = new_v

    # ---------------------------------------------------------------
    # Legacy on-path compression (track_only=False)
    # ---------------------------------------------------------------

    def _get_compressor(
        self, layer_idx: int, head_dim: int, device: torch.device
    ) -> TurboQuantV3:
        if layer_idx not in self._compressors:
            self._compressors[layer_idx] = TurboQuantV3(
                head_dim=head_dim,
                key_bits=self.config.key_bits,
                value_bits=self.config.value_bits,
                residual_window=0,  # we manage windowing here
                layer_idx=layer_idx,
                n_layers=self.n_layers,
                protected_layers=self.config.protected_layers,
                protected_bits=self.config.protected_bits,
                seed=self.config.seed,
                device=str(device),
            )
        return self._compressors[layer_idx]

    def _legacy_compress_overflow(
        self, layer_idx: int, new_total_len: int, template: torch.Tensor
    ) -> None:
        """Run actual compression on tokens that fell outside the residual window."""
        rw = self.config.residual_window
        if new_total_len <= rw:
            return
        # Count only chunks that are ACTUALLY compressed (have a blob). update()
        # pre-appends a span-only placeholder before calling us; including it here
        # would make want_compressed_upper <= already_compressed always true and
        # compression would never run.
        already_compressed = 0
        for c in self._chunks_k.get(layer_idx, []):
            if "blob" in c:
                already_compressed += c.get("span", 0)
        want_compressed_upper = new_total_len - rw
        if want_compressed_upper <= already_compressed:
            return

        start, end = already_compressed, want_compressed_upper
        overflow_k = self._buf_k[layer_idx][:, :, start:end, :]
        overflow_v = self._buf_v[layer_idx][:, :, start:end, :]

        comp = self._get_compressor(layer_idx, template.shape[-1], template.device)
        ck, cv = comp.compress_kv(overflow_k, overflow_v)
        span = end - start
        # Replace the lightweight placeholder added in update() with the full blob.
        if self._chunks_k[layer_idx] and "blob" not in self._chunks_k[layer_idx][-1]:
            self._chunks_k[layer_idx][-1] = {"blob": ck, "span": span}
            self._chunks_v[layer_idx][-1] = {"blob": cv, "span": span}
        else:
            self._chunks_k[layer_idx].append({"blob": ck, "span": span})
            self._chunks_v[layer_idx].append({"blob": cv, "span": span})

        # Write the lossy reconstruction back into the buffer ONCE, here, for the
        # span we just compressed (its source fp16 is still pristine). After this the
        # buffer holds [lossy compressed prefix][exact fp16 residual tail], so update()
        # can return a plain buffer slice — no per-step re-decompression. Each token is
        # decompressed exactly once → O(seq) total instead of O(seq^2).
        rk, rv = comp.decompress_kv(ck, cv)
        self._buf_k[layer_idx][:, :, start:end, :].copy_(
            rk.to(self._buf_k[layer_idx].dtype)
        )
        self._buf_v[layer_idx][:, :, start:end, :].copy_(
            rv.to(self._buf_v[layer_idx].dtype)
        )

    # ---------------------------------------------------------------
    # Public API — matches DynamicCache
    # ---------------------------------------------------------------

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new K/V and return full (K, V) slice views.

        Args:
            key_states:   (B, num_kv_heads, seq_new, head_dim)
            value_states: (B, num_kv_heads, seq_new, head_dim)
            layer_idx:    transformer layer index
            cache_kwargs: unused, kept for DynamicCache API compatibility
        Returns:
            (full_k, full_v) with shape (B, num_kv_heads, total_seq, head_dim).
        """
        s = self._cur_len.get(layer_idx, 0)
        n = key_states.shape[2]
        needed = s + n

        self._ensure_capacity(layer_idx, needed, template=key_states)

        # Single slice-write — no cat, no per-step dtype conversion.
        self._buf_k[layer_idx][:, :, s : s + n, :].copy_(key_states)
        self._buf_v[layer_idx][:, :, s : s + n, :].copy_(value_states)
        self._cur_len[layer_idx] = needed

        # Track virtual chunk spans for memory_report / test introspection.
        # One placeholder per overflow event (matches legacy chunk-per-overflow count).
        rw = self.config.residual_window
        if needed > rw:
            prev_compressed = sum(c.get("span", 0) for c in self._chunks_k[layer_idx])
            want_compressed = needed - rw
            new_span = want_compressed - prev_compressed
            if new_span > 0:
                self._chunks_k[layer_idx].append({"span": new_span})
                self._chunks_v[layer_idx].append({"span": new_span})

        if not self.config.track_only:
            self._legacy_compress_overflow(layer_idx, needed, template=key_states)

        # DynamicCache bookkeeping so transformers internals (e.g. len(cache))
        # keep reporting the right thing.
        self._ensure_layer_stubs(layer_idx)

        # In track_only=False mode, _legacy_compress_overflow has already written the
        # lossy reconstruction back into the buffer for the compressed prefix, so the
        # plain buffer slice already carries quantization loss (older tokens) + exact
        # fp16 (residual window). track_only=True leaves the buffer pristine.
        full_k = self._buf_k[layer_idx][:, :, :needed, :]
        full_v = self._buf_v[layer_idx][:, :, :needed, :]
        return full_k, full_v

    def _ensure_layer_stubs(self, layer_idx: int) -> None:
        """Ensure self.layers has enough entries for transformers internals."""
        from transformers.cache_utils import DynamicLayer

        while len(self.layers) <= layer_idx:
            self.layers.append(DynamicLayer())

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._cur_len.get(layer_idx, 0)

    @property
    def evicted_count(self) -> int:
        """Always 0 — TurboQuant compresses, it doesn't evict. RoPE positions stay correct."""
        return 0

    # ---------------------------------------------------------------
    # Reporting
    # ---------------------------------------------------------------

    def _layer_bit_widths(self, layer_idx: int) -> tuple[int, int]:
        """Mirror TurboQuantV3's layer-adaptive bit selection."""
        is_protected = layer_idx < self.config.protected_layers or layer_idx >= (
            self.n_layers - self.config.protected_layers
        )
        kb = self.config.protected_bits if is_protected else self.config.key_bits
        vb = self.config.protected_bits if is_protected else self.config.value_bits
        return min(kb, 8), min(vb, 8)

    def memory_report(self) -> dict:
        """Compressed vs fp16 memory accounting.

        Keys (preserved for backward compat with existing benchmarks + tests):
          compressed_bytes      — simulated compressed storage (analytical in
                                  track_only=True, actual in track_only=False)
          fp16_recent_bytes     — realized fp16 bytes the buffer actually stores
          total_bytes           — realized fp16 bytes (what the GPU actually pays)
          fp16_equivalent_bytes — fp16 cost of the full sequence, for ratio math
          compression_ratio     — fp16_equivalent / compressed (theoretical)
        """
        total_compressed_bytes = 0
        total_fp16_recent_bytes = 0
        fp16_equiv = 0

        for layer_idx, cur_len in self._cur_len.items():
            if cur_len == 0:
                continue
            buf = self._buf_k[layer_idx]
            B, H, _, D = buf.shape
            elem = buf.element_size()

            # Realized: the contents of the fp16 buffer, K + V.
            realized = 2 * B * H * cur_len * D * elem
            total_fp16_recent_bytes += realized
            fp16_equiv += 2 * B * H * cur_len * D * 2  # fp16 = 2 bytes

            kb, vb = self._layer_bit_widths(layer_idx)
            rw = min(self.config.residual_window, cur_len)
            compressed_tokens = max(cur_len - rw, 0)

            def _compressed_bytes_per_token(bits: int) -> int:
                indices_per_byte = 8 // bits
                idx_bytes = math.ceil(D / indices_per_byte)
                norm_bytes = 2  # fp16 vec norm
                return idx_bytes + norm_bytes

            # Simulated: what compression WOULD store = compressed chunks + fp16 residual.
            sim_k = (
                (compressed_tokens * _compressed_bytes_per_token(kb) + rw * D * 2)
                * B
                * H
            )
            sim_v = (
                (compressed_tokens * _compressed_bytes_per_token(vb) + rw * D * 2)
                * B
                * H
            )
            total_compressed_bytes += sim_k + sim_v

        total_bytes = total_fp16_recent_bytes
        compression_ratio = (
            fp16_equiv / total_compressed_bytes if total_compressed_bytes > 0 else 1.0
        )

        return {
            "compressed_bytes": total_compressed_bytes,
            "fp16_recent_bytes": total_fp16_recent_bytes,
            "total_bytes": total_bytes,
            "fp16_equivalent_bytes": fp16_equiv,
            "compression_ratio": compression_ratio,
            # Extra fields for parity with the VALL-E-X cache report:
            "realized_fp16_bytes": total_fp16_recent_bytes,
            "theoretical_compression_ratio": compression_ratio,
            "effective_compression_ratio": (
                fp16_equiv / total_fp16_recent_bytes
                if total_fp16_recent_bytes > 0
                else 1.0
            ),
        }

    def peak_vram_report(self, device) -> dict:
        """Thin wrapper over torch.cuda allocator stats — returns MB."""
        is_cuda = (isinstance(device, torch.device) and device.type == "cuda") or (
            isinstance(device, str) and device.startswith("cuda")
        )
        if not is_cuda:
            return {"peak_allocated_mb": 0.0, "current_allocated_mb": 0.0}
        return {
            "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
            "current_allocated_mb": torch.cuda.memory_allocated(device) / (1024**2),
        }
