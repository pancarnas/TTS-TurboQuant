"""Read-back semantics of TurboQuantKVCache for track_only True vs False.

These exercise the cache directly with synthetic tensors — no TTS model, no GPU —
so they pin the one behaviour that the full-model runs cannot cheaply verify:

  - track_only=True  -> update() returns the UNTOUCHED fp16 K/V (measurement only)
  - track_only=False -> update() returns a LOSSY reconstruction: tokens older than
    the residual window carry quantization error, the residual-window tail is exact.

The track_only=False case is the bug fix: previously the compressed blobs were
computed and thrown away, so the read-back was identical to baseline and the
downstream CER/temperature experiment could never see a compression effect.
"""

import importlib.util
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from turboquant.config import TurboQuantConfig

# Load the cache module by path (it lives under the qwen_tts package tree).
_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models",
    "Qwen3-TTS",
    "qwen_tts",
    "core",
    "models",
    "turboquant_kv_cache.py",
)
_spec = importlib.util.spec_from_file_location("turboquant_kv_cache", _CACHE_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
TurboQuantKVCache = _mod.TurboQuantKVCache


def _feed(cache, *, steps, step_len, n_heads=4, head_dim=64, seed=0):
    """Feed `steps` chunks of random K/V through layer 0; return (inputs, full_k, full_v)."""
    g = torch.Generator().manual_seed(seed)
    keys, vals = [], []
    full_k = full_v = None
    for _ in range(steps):
        k = torch.randn(1, n_heads, step_len, head_dim, generator=g)
        v = torch.randn(1, n_heads, step_len, head_dim, generator=g)
        keys.append(k)
        vals.append(v)
        full_k, full_v = cache.update(k, v, layer_idx=0)
    return torch.cat(keys, dim=2), torch.cat(vals, dim=2), full_k, full_v


def _config(track_only):
    # rw=16 so a few 8-token steps overflow it and force real compression.
    return TurboQuantConfig(
        key_bits=2,
        value_bits=2,
        residual_window=16,
        protected_layers=0,
        protected_bits=8,
        track_only=track_only,
    )


def test_track_only_true_returns_untouched_kv() -> None:
    cache = TurboQuantKVCache(_config(track_only=True), n_layers=1)
    in_k, in_v, full_k, full_v = _feed(cache, steps=6, step_len=8)
    # Measurement-only mode: read-back is bit-identical to what went in.
    assert torch.allclose(full_k, in_k, atol=0, rtol=0)
    assert torch.allclose(full_v, in_v, atol=0, rtol=0)


def test_track_only_false_reconstructs_lossily() -> None:
    cache = TurboQuantKVCache(_config(track_only=False), n_layers=1)
    in_k, in_v, full_k, full_v = _feed(cache, steps=6, step_len=8)
    total = in_k.shape[2]
    rw = 16

    # Shape must be preserved (drop-in for attention).
    assert full_k.shape == in_k.shape
    assert full_v.shape == in_v.shape

    # The residual-window tail (last rw tokens) stays exact fp16.
    assert torch.allclose(full_k[:, :, total - rw :, :], in_k[:, :, total - rw :, :])

    # The compressed prefix (older than rw) MUST differ from the original —
    # this is the whole point: generation now sees quantization loss.
    prefix_orig = in_k[:, :, : total - rw, :]
    prefix_read = full_k[:, :, : total - rw, :]
    assert not torch.allclose(prefix_read, prefix_orig, atol=1e-3), (
        "track_only=False read-back equals the original — compression is not "
        "reaching generation (the no-op bug)."
    )


def _prefix_key_mse(key_bits: int, value_bits: int) -> float:
    """Mean-squared reconstruction error on the compressed key prefix."""
    cfg = TurboQuantConfig(
        key_bits=key_bits,
        value_bits=value_bits,
        residual_window=16,
        protected_layers=0,
        protected_bits=8,
        track_only=False,
    )
    cache = TurboQuantKVCache(cfg, n_layers=1)
    in_k, _, full_k, _ = _feed(cache, steps=6, step_len=8)
    p = in_k.shape[2] - 16
    return (full_k[:, :, :p, :] - in_k[:, :, :p, :]).pow(2).mean().item()


def test_more_aggressive_compression_is_more_lossy() -> None:
    # The core research property: fewer bits => larger reconstruction error.
    assert _prefix_key_mse(4, 2) < _prefix_key_mse(2, 2)


def _config_rw0(track_only):
    """rw=0 — paper-faithful TurboQuant: every token quantized, no fp16 tail."""
    return TurboQuantConfig(
        key_bits=2,
        value_bits=2,
        residual_window=0,
        protected_layers=0,
        protected_bits=8,
        track_only=track_only,
    )


def test_rw0_quantizes_every_token() -> None:
    """At rw=0 the WHOLE sequence is lossy (no exact residual tail) — this is the
    behavior the thesis experiment depends on, so short sentences compress too."""
    cache = TurboQuantKVCache(_config_rw0(track_only=False), n_layers=1)
    in_k, in_v, full_k, full_v = _feed(cache, steps=6, step_len=8)

    assert full_k.shape == in_k.shape
    assert full_v.shape == in_v.shape
    # Even the most recent token must differ from its fp16 original — nothing is
    # kept exact at rw=0 (contrast the rw=16 test, which preserves the tail).
    assert not torch.allclose(full_k, in_k, atol=1e-3), (
        "rw=0 read-back equals the original — compression is not reaching "
        "generation; short sentences would be byte-identical to baseline."
    )


def test_rw0_track_only_true_still_measurement_only() -> None:
    """rw=0 must not corrupt the measurement-only path: track_only=True still
    returns untouched fp16 (the buffer stays pristine)."""
    cache = TurboQuantKVCache(_config_rw0(track_only=True), n_layers=1)
    in_k, in_v, full_k, full_v = _feed(cache, steps=6, step_len=8)
    assert torch.allclose(full_k, in_k, atol=0, rtol=0)
    assert torch.allclose(full_v, in_v, atol=0, rtol=0)


def test_rw0_single_token_prefill_edge() -> None:
    """S=1 prefill at rw=0 (empty residual slice) must not break — the
    compressor handles a one-token sequence and the read-back stays shaped."""
    cache = TurboQuantKVCache(_config_rw0(track_only=False), n_layers=1)
    in_k, in_v, full_k, full_v = _feed(cache, steps=1, step_len=1)
    assert full_k.shape == in_k.shape == (1, 4, 1, 64)
    # One step further (decode token) must also work without error.
    g = torch.Generator().manual_seed(1)
    k = torch.randn(1, 4, 1, 64, generator=g)
    v = torch.randn(1, 4, 1, 64, generator=g)
    fk, fv = cache.update(k, v, layer_idx=0)
    assert fk.shape == (1, 4, 2, 64)


def test_decompress_is_called_once_per_chunk_not_per_step() -> None:
    """Guards the O(seq) write-back path: each compressed span is decompressed
    exactly once (at compression time), NOT re-decompressed on every read. A
    regression to per-step reconstruction would make this scale with steps."""
    cfg = TurboQuantConfig(
        key_bits=2,
        value_bits=2,
        residual_window=16,
        protected_layers=0,
        protected_bits=8,
        track_only=False,
    )
    cache = TurboQuantKVCache(cfg, n_layers=1)

    calls = {"n": 0}
    orig_get = cache._get_compressor

    def counting_get(layer_idx, head_dim, device):
        comp = orig_get(layer_idx, head_dim, device)
        if not getattr(comp, "_wrapped", False):
            comp._wrapped = True
            real = comp.decompress_kv

            def wrapped(*a, **k):
                calls["n"] += 1
                return real(*a, **k)

            comp.decompress_kv = wrapped
        return comp

    cache._get_compressor = counting_get
    _feed(cache, steps=6, step_len=8)  # 48 tokens, rw=16 → 4 overflow chunks
    # One decompress per overflow chunk, never per read/step.
    assert calls["n"] <= 6, f"decompress called {calls['n']}x — not O(seq)"
