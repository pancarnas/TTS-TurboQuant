"""TurboQuant-compressed KV cache for Qwen3-TTS.

Drop-in replacement for HuggingFace's DynamicCache that tracks TurboQuant's
theoretical compression metrics while storing K/V in preallocated fp16 buffers.

Phase 1 rewrite (2026-04-22, see feature/valle EXPERIMENT_LOG.md): replaced the
list-of-tensors + torch.cat-per-step pattern (plus per-step CPU<->GPU round
trip on the decompressed prefix) with grow-on-doubling preallocated buffers,
one per layer, slice-written in place. Compression is off-path by default via
config.track_only=True; legacy on-path compression is still available for
reconstruction-quality A/B tests via track_only=False.

Key-fix experiments (docs/keyfix_experiment.md):
  * config.key_group > 0  — per-group min-max key scaling (TurboQuantV3 does
    the work; this cache only rounds compressed spans to whole groups).
  * config.pre_rope_keys  — quantize keys BEFORE RoPE. The attention layer
    hands us the raw rotary cos/sin in ``cache_kwargs`` every step; we keep a
    per-position copy, undo RoPE on the span that ages out of the residual
    window, quantize->dequantize, re-apply RoPE and write back. Attention
    still reads post-RoPE keys exactly as before.

Typical usage:
    config = TurboQuantConfig(key_bits=4, value_bits=2)
    cache = TurboQuantKVCache(config, n_layers=20, rope_scaling=talker_config.rope_scaling)
    # pass as past_key_values to model.generate()
"""

from typing import Optional

import torch
from transformers.cache_utils import DynamicCache

from turboquant.config import TurboQuantConfig
from turboquant.compressors_v3 import TurboQuantV3, compressed_bytes_per_token


# Initial buffer capacity (tokens). Grows by doubling on overflow.
_INITIAL_CAPACITY = 256


# ---------------------------------------------------------------------------
# RoPE helpers (self-contained mirror of modeling_qwen3_tts.apply_*rotary*)
# ---------------------------------------------------------------------------


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def prepare_rope(cos: torch.Tensor, sin: torch.Tensor,
                 mrope_section=None, interleaved: bool = False):
    """Raw rotary tables -> per-head-broadcastable (B, 1, S, D) tables.

    Mirrors the pre-processing inside ``apply_multimodal_rotary_pos_emb``
    (talker: cos/sin of shape (3, B, S, D) with an mrope section split) and
    ``apply_rotary_pos_emb`` (code predictor: plain (B, S, D)). ``cos.dim()``
    decides which; ``mrope_section`` is required for the 4-D case.
    """
    if cos.dim() == 3:  # plain RoPE
        return cos.unsqueeze(1), sin.unsqueeze(1)
    if cos.dim() != 4:
        raise ValueError(f"unexpected rotary table shape {tuple(cos.shape)}")
    if mrope_section is None:
        raise ValueError("mrope_section is required for 4-D (multimodal) rotary tables")
    mrope_section = list(mrope_section)
    if interleaved:
        modality_num = len(mrope_section)

        def _interleave(x):
            x_t = x[0].clone()
            for i, n in enumerate(mrope_section[1:], 1):
                x_t[..., i : n * modality_num : modality_num] = x[i, ..., i : n * modality_num : modality_num]
            return x_t

        dim = cos.shape[-1]
        cos_p = torch.cat([_interleave(cos[..., : dim // 2])] * 2, dim=-1)
        sin_p = torch.cat([_interleave(sin[..., : dim // 2])] * 2, dim=-1)
    else:
        sec = mrope_section * 2
        cos_p = torch.cat([m[i % 3] for i, m in enumerate(cos.split(sec, dim=-1))], dim=-1)
        sin_p = torch.cat([m[i % 3] for i, m in enumerate(sin.split(sec, dim=-1))], dim=-1)
    return cos_p.unsqueeze(1), sin_p.unsqueeze(1)


def rope_keys(k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
              mrope_section=None, interleaved: bool = False,
              inverse: bool = False) -> torch.Tensor:
    """Apply (or, with ``inverse=True``, exactly undo) RoPE on keys (B, H, S, D).

    ``cos``/``sin`` are the RAW tables for exactly these S positions, as the
    attention layer received them from the rotary module. The inverse is the
    same formula with the sign of ``sin`` flipped (rotation by -theta), which
    is exact because every table is the concatenation of two identical halves.
    """
    cos_p, sin_p = prepare_rope(cos, sin, mrope_section, interleaved)
    if inverse:
        sin_p = -sin_p
    cos_p = cos_p.to(k.dtype)
    sin_p = sin_p.to(k.dtype)
    return (k * cos_p) + (_rotate_half(k) * sin_p)


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

    def __init__(self, config: TurboQuantConfig, n_layers: int = 20,
                 rope_scaling: Optional[dict] = None):
        super().__init__()
        self.config = config
        self.n_layers = n_layers
        # mrope section / interleave flags for pre-RoPE key quantization
        # (talker_config.rope_scaling); None = plain 1-D RoPE tables.
        self.rope_scaling = rope_scaling

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
        # Explicit upper bound of the ACTUALLY compressed prefix per layer
        # (track_only=False). Independent of the placeholder bookkeeping so
        # group-rounded spans can never desynchronise it.
        self._compressed_upto: dict[int, int] = {}

        # Lazy per-layer compressor (only instantiated when track_only=False).
        self._compressors: dict[int, TurboQuantV3] = {}

        # Per-position raw rotary tables (pre_rope_keys only). Shared by all
        # layers — the rotary module emits one table per step — so they are
        # written once per step by whichever layer updates first.
        self._cos_buf: Optional[torch.Tensor] = None
        self._sin_buf: Optional[torch.Tensor] = None
        self._rope_len: int = 0

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
            self._compressed_upto[layer_idx] = 0
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
    # Rotary table buffer (pre_rope_keys)
    # ---------------------------------------------------------------

    def _record_rope(self, start: int, cos: torch.Tensor, sin: torch.Tensor) -> None:
        """Store the raw cos/sin for positions [start, start + S_new)."""
        n = cos.shape[-2]
        needed = start + n
        if self._rope_len >= needed:
            return  # another layer already wrote this step's tables
        if self._rope_len != start:
            raise RuntimeError(
                f"pre_rope_keys: rotary table gap (have {self._rope_len} positions, "
                f"step starts at {start}) — cache_kwargs must carry cos/sin every step"
            )
        lead = cos.shape[:-2]
        D = cos.shape[-1]
        if self._cos_buf is None or self._cos_buf.shape[-2] < needed:
            cap = max(_INITIAL_CAPACITY, self._cos_buf.shape[-2] if self._cos_buf is not None else 0)
            while cap < needed:
                cap *= 2
            new_cos = torch.empty(*lead, cap, D, dtype=cos.dtype, device=cos.device)
            new_sin = torch.empty(*lead, cap, D, dtype=sin.dtype, device=sin.device)
            if self._cos_buf is not None and self._rope_len > 0:
                new_cos[..., : self._rope_len, :].copy_(self._cos_buf[..., : self._rope_len, :])
                new_sin[..., : self._rope_len, :].copy_(self._sin_buf[..., : self._rope_len, :])
            self._cos_buf, self._sin_buf = new_cos, new_sin
        self._cos_buf[..., start:needed, :].copy_(cos)
        self._sin_buf[..., start:needed, :].copy_(sin)
        self._rope_len = needed

    def _rope_slice(self, start: int, end: int):
        if self._rope_len < end:
            raise RuntimeError(
                f"pre_rope_keys: no rotary tables for positions [{start}, {end}) "
                f"(have {self._rope_len})"
            )
        return self._cos_buf[..., start:end, :], self._sin_buf[..., start:end, :]

    def _rope_kwargs(self) -> dict:
        rs = self.rope_scaling or {}
        return {
            "mrope_section": rs.get("mrope_section"),
            "interleaved": bool(rs.get("interleaved", False)),
        }

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
                key_group=self.config.key_group,
            )
        return self._compressors[layer_idx]

    def _legacy_compress_overflow(
        self, layer_idx: int, new_total_len: int, template: torch.Tensor
    ) -> None:
        """Run actual compression on tokens that fell outside the residual window.

        With ``config.key_group`` > 0 the span end is rounded DOWN to a multiple
        of the group so every compressed block is a whole group aligned to
        position 0; the partial tail stays fp16 until its group fills.
        """
        rw = self.config.residual_window
        if new_total_len <= rw:
            return
        already_compressed = self._compressed_upto.get(layer_idx, 0)
        want_compressed_upper = new_total_len - rw
        g = self.config.key_group
        if g > 0:
            want_compressed_upper = already_compressed + ((want_compressed_upper - already_compressed) // g) * g
        if want_compressed_upper <= already_compressed:
            return

        start, end = already_compressed, want_compressed_upper
        overflow_k = self._buf_k[layer_idx][:, :, start:end, :]
        overflow_v = self._buf_v[layer_idx][:, :, start:end, :]

        rope_kw = None
        if self.config.pre_rope_keys:
            # Undo RoPE on the aged-out span so the quantizer sees the
            # position-free keys; RoPE is re-applied to the reconstruction.
            cos, sin = self._rope_slice(start, end)
            rope_kw = self._rope_kwargs()
            overflow_k = rope_keys(overflow_k, cos, sin, inverse=True, **rope_kw)

        comp = self._get_compressor(layer_idx, template.shape[-1], template.device)
        ck, cv = comp.compress_kv(overflow_k, overflow_v)
        span = end - start
        # Replace the lightweight placeholder added in update() when it covers
        # exactly this span; otherwise (group-rounded spans) append.
        chunks_k, chunks_v = self._chunks_k[layer_idx], self._chunks_v[layer_idx]
        if chunks_k and "blob" not in chunks_k[-1] and chunks_k[-1].get("span") == span:
            chunks_k[-1] = {"blob": ck, "span": span}
            chunks_v[-1] = {"blob": cv, "span": span}
        else:
            chunks_k.append({"blob": ck, "span": span})
            chunks_v.append({"blob": cv, "span": span})
        self._compressed_upto[layer_idx] = end

        # Write the lossy reconstruction back into the buffer ONCE, here, for the
        # span we just compressed (its source fp16 is still pristine). After this the
        # buffer holds [lossy compressed prefix][exact fp16 residual tail], so update()
        # can return a plain buffer slice — no per-step re-decompression. Each token is
        # decompressed exactly once → O(seq) total instead of O(seq^2).
        rk, rv = comp.decompress_kv(ck, cv)
        if rope_kw is not None:
            rk = rope_keys(rk.to(self._buf_k[layer_idx].dtype), cos, sin, inverse=False, **rope_kw)
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
            cache_kwargs: ``{"cos", "sin", "cache_position"}`` from the attention
                          layer; only read when ``config.pre_rope_keys`` is set.
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

        if self.config.pre_rope_keys:
            if not cache_kwargs or "cos" not in cache_kwargs or "sin" not in cache_kwargs:
                raise RuntimeError(
                    "pre_rope_keys=True needs cache_kwargs['cos'/'sin'] from the attention layer"
                )
            self._record_rope(s, cache_kwargs["cos"], cache_kwargs["sin"])

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

            # Simulated: what compression WOULD store = compressed chunks + fp16
            # residual. Shared formula (turboquant.compressors_v3); keys carry
            # the group min/scale metadata when key_group > 0.
            sim_k = (
                (compressed_tokens * compressed_bytes_per_token(D, kb, self.config.key_group) + rw * D * 2)
                * B
                * H
            )
            sim_v = (
                (compressed_tokens * compressed_bytes_per_token(D, vb) + rw * D * 2)
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
