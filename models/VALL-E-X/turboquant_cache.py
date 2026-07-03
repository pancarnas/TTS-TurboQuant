"""TurboQuant KV cache compression for VALL-E-X.

VALL-E-X uses a tuple-based KV cache: tuple of 12 (key, value) pairs, each
(B, num_heads=16, seq_len, head_dim=64).

Phase 1 (2026-04-21) rewrite: preallocated per-layer buffers. A single grow-on-
doubling fp16 tensor per layer replaces the old list-of-tensors + torch.cat-per-
step pattern. On an L4 this cuts ~30k torch.cat / ~65k aten::to / ~13k
cudaStreamSynchronize events per inference and eliminates the 7x latency
regression observed vs baseline.

Two modes:
  - config.track_only=True (default) — fp16 buffers only, compression metrics
    are computed analytically from cur_len + config. Fast path for inference.
  - config.track_only=False — LOSSY quality-experiment path: tokens that age
    out of the residual window are compressed once via TurboQuantV3 and the
    quantize->dequantize reconstruction is written back into the fp16 buffer,
    so attention reads exactly what a deployed compressed cache would serve.
    The compressed blobs are also retained for the memory report. Storage
    stays fp16-realized (VALL-E-X's softmax attention reads fp16 K/V), so the
    memory savings remain theoretical — but the QUALITY effect is real.
"""

import math
import sys
import os
from typing import Optional

import torch

# Make the shared `turboquant` package importable when this file is loaded via
# sys.path injection from the benchmark script (and not via a pip install).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from turboquant.config import TurboQuantConfig
from turboquant.compressors_v3 import TurboQuantV3


# Initial buffer capacity (tokens). Grows by doubling on overflow.
_INITIAL_CAPACITY = 256


class TurboQuantValleCache:
    """Preallocated-buffer KV cache manager for VALL-E-X.

    Public API (unchanged from Phase 0):
        cache = TurboQuantValleCache(config, n_layers=12)
        full_k, full_v = cache.update(layer_idx, new_key, new_value)
        cache.get_seq_length(layer_idx)
        cache.memory_report()
        cache.peak_vram_report(device)
    """

    def __init__(self, config: TurboQuantConfig, n_layers: int = 12):
        self.config = config
        self.n_layers = n_layers

        # Preallocated per-layer fp16 buffers (the realized storage the
        # attention layer actually reads).
        self._buf_k: dict[int, torch.Tensor] = {}
        self._buf_v: dict[int, torch.Tensor] = {}
        self._cur_len: dict[int, int] = {}

        # Legacy off-path compression state (only populated when
        # config.track_only=False).
        self._compressors: dict[int, TurboQuantV3] = {}
        self._compressed_chunks_k: dict[int, list[dict]] = {}
        self._compressed_chunks_v: dict[int, list[dict]] = {}

    # ---------------------------------------------------------------
    # Buffer management
    # ---------------------------------------------------------------

    def _ensure_capacity(self, layer_idx: int, needed_len: int,
                         template: torch.Tensor) -> None:
        """Ensure the per-layer buffer has capacity for `needed_len` tokens."""
        B, H, _, D = template.shape
        dtype = template.dtype
        device = template.device

        if layer_idx not in self._buf_k:
            cap = max(_INITIAL_CAPACITY, needed_len)
            self._buf_k[layer_idx] = torch.empty(B, H, cap, D, dtype=dtype, device=device)
            self._buf_v[layer_idx] = torch.empty(B, H, cap, D, dtype=dtype, device=device)
            self._cur_len[layer_idx] = 0
            return

        cur_cap = self._buf_k[layer_idx].shape[2]
        if needed_len <= cur_cap:
            return

        # Grow by doubling. One cat per doubling step, not per AR step —
        # O(log n) cats total instead of O(n).
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
    # Legacy off-path compression (only when track_only=False)
    # ---------------------------------------------------------------

    def _get_compressor(self, layer_idx: int, head_dim: int,
                        device: torch.device) -> TurboQuantV3:
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

    def _legacy_compress_overflow(self, layer_idx: int, new_total_len: int,
                                  new_key: torch.Tensor,
                                  new_value: torch.Tensor) -> None:
        """When track_only=False, compress tokens that fall out of the residual
        window, write the lossy reconstruction back into the fp16 buffer (so
        attention reads quantized K/V, as deployment would), and keep the
        compressed blobs around for memory_report. Each token is quantized
        exactly once — at the step it ages out of the window — and the
        reconstruction persists in the buffer from then on.
        """
        rw = self.config.residual_window
        if new_total_len <= rw:
            return

        # We need to compress any tokens in [cur_compressed_upper, new_total_len - rw).
        # Track upper bound via length of compressed_chunks (in tokens) — each
        # chunk records its span.
        already_compressed = 0
        chunks = self._compressed_chunks_k.get(layer_idx, [])
        for c in chunks:
            already_compressed += c["span"]

        want_compressed_upper = new_total_len - rw
        if want_compressed_upper <= already_compressed:
            return

        start = already_compressed
        end = want_compressed_upper
        overflow_k = self._buf_k[layer_idx][:, :, start:end, :]
        overflow_v = self._buf_v[layer_idx][:, :, start:end, :]

        comp = self._get_compressor(layer_idx, new_key.shape[-1], new_key.device)
        ck, cv = comp.compress_kv(overflow_k, overflow_v)

        # Lossy write-back: attention must read what a deployed compressed
        # cache would serve, so overwrite the aged-out span with the
        # quantize->dequantize reconstruction. (The chunk compressor has
        # residual_window=0, so decompress_kv returns exactly this span.)
        rec_k, rec_v = comp.decompress_kv(ck, cv)
        buf_dtype = self._buf_k[layer_idx].dtype
        self._buf_k[layer_idx][:, :, start:end, :].copy_(rec_k.to(buf_dtype))
        self._buf_v[layer_idx][:, :, start:end, :].copy_(rec_v.to(buf_dtype))

        self._compressed_chunks_k.setdefault(layer_idx, []).append(
            {"blob": ck, "span": end - start}
        )
        self._compressed_chunks_v.setdefault(layer_idx, []).append(
            {"blob": cv, "span": end - start}
        )

    # ---------------------------------------------------------------
    # Public update — one slice write per step, O(1) per step amortized.
    # ---------------------------------------------------------------

    def update(self, layer_idx: int, new_key: torch.Tensor,
               new_value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new K/V tokens and return full (K, V) slice views.

        new_key / new_value: (B, H, seq_new, D). Only the NEW tokens this step.
        Returns: contiguous slice views of shape (B, H, total_seq, D).
        """
        s = self._cur_len.get(layer_idx, 0)
        n = new_key.shape[2]
        needed = s + n

        self._ensure_capacity(layer_idx, needed, template=new_key)

        # Slice write — no cat, no dtype conversion, no intermediate tensor.
        self._buf_k[layer_idx][:, :, s:s + n, :].copy_(new_key)
        self._buf_v[layer_idx][:, :, s:s + n, :].copy_(new_value)
        self._cur_len[layer_idx] = needed

        if not self.config.track_only:
            self._legacy_compress_overflow(layer_idx, needed, new_key, new_value)

        full_k = self._buf_k[layer_idx][:, :, :needed, :]
        full_v = self._buf_v[layer_idx][:, :, :needed, :]
        return full_k, full_v

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._cur_len.get(layer_idx, 0)

    # ---------------------------------------------------------------
    # Reporting
    # ---------------------------------------------------------------

    def _layer_bit_widths(self, layer_idx: int) -> tuple[int, int]:
        """Mirror TurboQuantV3's layer-adaptive bit selection."""
        is_protected = (
            layer_idx < self.config.protected_layers
            or layer_idx >= (self.n_layers - self.config.protected_layers)
        )
        kb = self.config.protected_bits if is_protected else self.config.key_bits
        vb = self.config.protected_bits if is_protected else self.config.value_bits
        return min(kb, 8), min(vb, 8)

    def memory_report(self) -> dict:
        """Memory accounting.

        Keys:
          realized_fp16_bytes: VRAM the cache actually uses (fp16 buffers).
          buffer_capacity_bytes: VRAM reserved by preallocated buffers (>= realized).
          simulated_compressed_bytes: what TurboQuant WOULD use at current bit-widths
              (compressed chunks beyond the residual window + fp16 for recent tokens).
          fp16_equivalent_bytes: fp16 cost of the same total K/V.
          theoretical_compression_ratio: fp16_equivalent_bytes / simulated_compressed_bytes.
          effective_compression_ratio: fp16_equivalent_bytes / realized_fp16_bytes.
              In track_only mode this is always 1.0 (we store fp16). The
              theoretical ratio is what compression WOULD deliver if a
              compression-aware attention kernel were available.
          compression_ratio: alias for theoretical_compression_ratio (backward compat).
        """
        realized = 0
        buffer_cap = 0
        simulated = 0
        fp16_equiv = 0

        for layer_idx, cur_len in self._cur_len.items():
            buf = self._buf_k[layer_idx]
            B, H, cap, D = buf.shape
            elem_size = buf.element_size()

            realized += 2 * B * H * cur_len * D * elem_size  # K + V
            buffer_cap += 2 * B * H * cap * D * elem_size
            fp16_equiv += 2 * B * H * cur_len * D * 2  # K + V at fp16

            kb, vb = self._layer_bit_widths(layer_idx)
            rw = min(self.config.residual_window, cur_len)
            compressed_tokens = max(cur_len - rw, 0)

            def _compressed_bytes_per_token(bits: int) -> int:
                indices_per_byte = 8 // bits
                idx_bytes = math.ceil(D / indices_per_byte)
                norm_bytes = 2  # fp16 vec norm
                return idx_bytes + norm_bytes

            sim_k = (compressed_tokens * _compressed_bytes_per_token(kb)
                     + rw * D * 2) * B * H
            sim_v = (compressed_tokens * _compressed_bytes_per_token(vb)
                     + rw * D * 2) * B * H
            simulated += sim_k + sim_v

        theoretical_ratio = fp16_equiv / simulated if simulated > 0 else 1.0
        effective_ratio = fp16_equiv / realized if realized > 0 else 1.0

        # Compressed-chunk-on-GPU size if track_only=False, for transparency.
        compressed_chunks_bytes = 0
        for chunks in list(self._compressed_chunks_k.values()) + list(self._compressed_chunks_v.values()):
            for c in chunks:
                blob = c.get("blob")
                if isinstance(blob, dict):
                    for v in blob.values():
                        if isinstance(v, torch.Tensor):
                            compressed_chunks_bytes += v.nelement() * v.element_size()

        return {
            "realized_fp16_bytes": realized,
            "buffer_capacity_bytes": buffer_cap,
            "simulated_compressed_bytes": simulated,
            "compressed_chunks_on_gpu_bytes": compressed_chunks_bytes,
            "fp16_equivalent_bytes": fp16_equiv,
            "theoretical_compression_ratio": theoretical_ratio,
            "effective_compression_ratio": effective_ratio,
            # Backward-compat keys for the benchmark that expected Phase-0 field names.
            "compression_ratio": theoretical_ratio,
            "compressed_bytes": simulated,
            "fp16_recent_bytes": realized,  # all realized bytes are fp16 here
            "decompressed_prefix_bytes": 0,  # no longer a separate concept
            "total_bytes": realized,
        }

    def peak_vram_report(self, device) -> dict:
        """Thin wrapper over torch.cuda allocator stats — returns MB."""
        if not (isinstance(device, torch.device) and device.type == "cuda") and \
           not (isinstance(device, str) and device.startswith("cuda")):
            return {"peak_allocated_mb": 0.0, "current_allocated_mb": 0.0}
        return {
            "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024 ** 2),
            "current_allocated_mb": torch.cuda.memory_allocated(device) / (1024 ** 2),
        }
