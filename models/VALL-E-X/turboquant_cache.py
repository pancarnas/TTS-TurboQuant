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


class DivergenceRecorder:
    """Per-(layer, position) attention-divergence stats, quantized vs exact.

    Attach to a TurboQuantValleCache (``cache.recorder = rec``): the cache
    then keeps a shadow un-quantized K/V copy, and the attention hook in
    modules/activation.py calls ``observe()`` each decode step with the
    current query plus both K/V versions. Metric names and semantics mirror
    the Qwen divergence experiment (kv_attn_divergence_experiment.py) so
    tools/analyze_divergence.py reads both CSVs:

      attn_js / attn_tv / attn_top1 / attn_dentropy — divergence between the
        attention distributions computed with quantized vs exact keys (last
        query only, i.e. the token being decoded), averaged over heads;
      out_cos — cosine of the two attention *outputs* (att @ V);
      cos_k/cos_v, relmse_k/relmse_v — K/V reconstruction on the compressed
        prefix only (the last ``rw`` tokens are bit-exact and would dilute
        the error as S grows).

    ``rows`` collects [layer, pos, <metrics...>]; the driver script prepends
    sentence/config columns when writing the CSV.
    """

    METRICS = [
        "attn_js",
        "attn_tv",
        "attn_top1",
        "attn_dentropy",
        "out_cos",
        "cos_k",
        "cos_v",
        "relmse_k",
        "relmse_v",
    ]

    def __init__(self, residual_window: int, step_stride: int = 1):
        self.rw = residual_window
        self.stride = max(int(step_stride), 1)
        self.rows: list[list] = []

    @staticmethod
    def _entropy(p: torch.Tensor) -> torch.Tensor:
        return -(p * (p + 1e-12).log()).sum(-1)

    @torch.no_grad()
    def observe(self, layer_idx: int, q: torch.Tensor,
                k_q: torch.Tensor, v_q: torch.Tensor,
                k_e: torch.Tensor, v_e: torch.Tensor) -> None:
        """q: (B,H,T,D) this step's queries; k/v_(q|e): (B,H,S,D) full cache,
        quantized and exact. Stats are for the LAST query position only."""
        S = k_q.shape[2]
        pos = S - 1
        if pos % self.stride:
            return

        ql = q[:, :, -1:, :].float()
        scale = 1.0 / math.sqrt(q.shape[-1])
        att_q = torch.softmax((ql @ k_q.float().transpose(-2, -1)) * scale, dim=-1)
        att_e = torch.softmax((ql @ k_e.float().transpose(-2, -1)) * scale, dim=-1)

        m = 0.5 * (att_q + att_e)
        kl_qm = (att_q * ((att_q + 1e-12).log() - (m + 1e-12).log())).sum(-1)
        kl_em = (att_e * ((att_e + 1e-12).log() - (m + 1e-12).log())).sum(-1)
        attn_js = float((0.5 * (kl_qm + kl_em)).mean())
        attn_tv = float((0.5 * (att_q - att_e).abs().sum(-1)).mean())
        attn_top1 = float(
            (att_q.argmax(-1) == att_e.argmax(-1)).float().mean()
        )
        attn_dentropy = float((self._entropy(att_q) - self._entropy(att_e)).mean())

        out_q = (att_q @ v_q.float()).flatten()
        out_e = (att_e @ v_e.float()).flatten()
        out_cos = float(
            torch.nn.functional.cosine_similarity(out_q, out_e, dim=0)
        )

        split = max(S - self.rw, 0)
        if split > 0:
            kq, ke = k_q[:, :, :split, :].float(), k_e[:, :, :split, :].float()
            vq, ve = v_q[:, :, :split, :].float(), v_e[:, :, :split, :].float()
            cos_k = float(
                torch.nn.functional.cosine_similarity(kq, ke, dim=-1).mean()
            )
            cos_v = float(
                torch.nn.functional.cosine_similarity(vq, ve, dim=-1).mean()
            )
            relmse_k = float((kq - ke).pow(2).sum() / (ke.pow(2).sum() + 1e-12))
            relmse_v = float((vq - ve).pow(2).sum() / (ve.pow(2).sum() + 1e-12))
        else:
            cos_k = cos_v = relmse_k = relmse_v = ""

        self.rows.append(
            [layer_idx, pos, attn_js, attn_tv, attn_top1, attn_dentropy,
             out_cos, cos_k, cos_v, relmse_k, relmse_v]
        )


class NARQuantizer:
    """Stateless per-call K/V quantize->dequantize for the NAR decoder.

    Duck-typed to the attention hook in modules/activation.py
    (multi_head_attention_forward): exposes ``update(layer_idx, k, v)``.
    Unlike TurboQuantValleCache there is no storage — each NAR stage is a
    full-sequence bidirectional pass, so "compression" means quantizing the
    K/V that pass computes: tokens ``[0 : S-rw]`` lossy, the last ``rw``
    tokens kept fp16. This is a robustness ablation, not a memory saving
    (NAR caches nothing between stages).
    """

    def __init__(self, config: TurboQuantConfig, n_layers: int = 12):
        self.config = config
        self.n_layers = n_layers
        self._compressors: dict[int, TurboQuantV3] = {}

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

    # Tokens per compress_kv call. The MSE quantizer's temp buffers scale
    # with the tokens it sees at once ((B*H*S, D, n_levels) diffs), which
    # OOMs on multi-minute NAR passes; chunking is EXACT (quantization is
    # per-vector), it only bounds the transient memory.
    _CHUNK_TOKENS = 2048

    @torch.no_grad()
    def update(self, layer_idx: int, k: torch.Tensor,
               v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize->dequantize all but the last residual_window tokens.

        k / v: (B, H, S, D) — the FULL sequence of one NAR pass, not an
        incremental step. Returns same-shape tensors.
        """
        S = k.shape[2]
        split = max(S - self.config.residual_window, 0)
        if split == 0:
            return k, v
        comp = self._get_compressor(layer_idx, k.shape[-1], k.device)
        rec_k, rec_v = [], []
        for s0 in range(0, split, self._CHUNK_TOKENS):
            s1 = min(s0 + self._CHUNK_TOKENS, split)
            ck, cv = comp.compress_kv(k[:, :, s0:s1, :], v[:, :, s0:s1, :])
            rk, rv = comp.decompress_kv(ck, cv)
            rec_k.append(rk.to(k.dtype))
            rec_v.append(rv.to(v.dtype))
        k = torch.cat(rec_k + [k[:, :, split:, :]], dim=2)
        v = torch.cat(rec_v + [v[:, :, split:, :]], dim=2)
        return k, v


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

        # Divergence recording (diagnostics only): when a DivergenceRecorder
        # is attached, a shadow UN-quantized K/V copy is kept per layer so the
        # attention hook can compare quantized vs exact attention.
        self.recorder = None
        self._exact_k: dict[int, torch.Tensor] = {}
        self._exact_v: dict[int, torch.Tensor] = {}

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

        # Shadow exact copy (diagnostics): written BEFORE the lossy path can
        # overwrite the main buffer, never quantized.
        if self.recorder is not None:
            self._ensure_exact_capacity(layer_idx, needed, template=new_key)
            self._exact_k[layer_idx][:, :, s:s + n, :].copy_(new_key)
            self._exact_v[layer_idx][:, :, s:s + n, :].copy_(new_value)

        if not self.config.track_only:
            self._legacy_compress_overflow(layer_idx, needed, new_key, new_value)

        full_k = self._buf_k[layer_idx][:, :, :needed, :]
        full_v = self._buf_v[layer_idx][:, :, :needed, :]
        return full_k, full_v

    def _ensure_exact_capacity(self, layer_idx: int, needed_len: int,
                               template: torch.Tensor) -> None:
        B, H, _, D = template.shape
        buf = self._exact_k.get(layer_idx)
        cap = buf.shape[2] if buf is not None else 0
        if cap >= needed_len:
            return
        new_cap = max(_INITIAL_CAPACITY, cap)
        while new_cap < needed_len:
            new_cap *= 2
        for store in (self._exact_k, self._exact_v):
            new = torch.empty(B, H, new_cap, D, dtype=template.dtype,
                              device=template.device)
            old = store.get(layer_idx)
            if old is not None:
                s = min(self._cur_len.get(layer_idx, 0), old.shape[2])
                new[:, :, :s, :].copy_(old[:, :, :s, :])
            store[layer_idx] = new

    def exact_kv(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Un-quantized shadow K/V views (recorder mode only)."""
        s = self._cur_len.get(layer_idx, 0)
        return (
            self._exact_k[layer_idx][:, :, :s, :],
            self._exact_v[layer_idx][:, :, :s, :],
        )

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
