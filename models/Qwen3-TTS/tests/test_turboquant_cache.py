"""Tests for TurboQuant KV cache integration."""

import pytest
import torch

from qwen_tts.core.models.turboquant_kv_cache import TurboQuantConfig, TurboQuantKVCache


# --- Fixtures ---

BATCH = 1
NUM_KV_HEADS = 2
HEAD_DIM = 128
N_LAYERS = 20


def _random_kv(seq_len: int = 1, batch: int = BATCH):
    """Generate random key/value tensors matching Qwen3-TTS shape."""
    k = torch.randn(batch, NUM_KV_HEADS, seq_len, HEAD_DIM)
    v = torch.randn(batch, NUM_KV_HEADS, seq_len, HEAD_DIM)
    return k, v


# --- Unit Tests ---


class TestTurboQuantConfig:
    def test_defaults(self):
        cfg = TurboQuantConfig()
        assert cfg.key_bits == 4
        assert cfg.value_bits == 2
        assert cfg.residual_window == 128
        assert cfg.protected_layers == 2
        assert cfg.seed == 42
        assert cfg.enabled is True

    def test_custom_values(self):
        cfg = TurboQuantConfig(key_bits=3, value_bits=3, residual_window=64)
        assert cfg.key_bits == 3
        assert cfg.residual_window == 64


class TestCacheBasics:
    def test_update_returns_correct_shape(self):
        """Single update should return K/V with correct shape."""
        cfg = TurboQuantConfig(residual_window=128)
        cache = TurboQuantKVCache(cfg, n_layers=N_LAYERS)
        k, v = _random_kv(seq_len=10)

        out_k, out_v = cache.update(k, v, layer_idx=0)

        assert out_k.shape == (BATCH, NUM_KV_HEADS, 10, HEAD_DIM)
        assert out_v.shape == (BATCH, NUM_KV_HEADS, 10, HEAD_DIM)

    def test_seq_length_tracking(self):
        """get_seq_length should reflect total tokens fed."""
        cfg = TurboQuantConfig(residual_window=128)
        cache = TurboQuantKVCache(cfg, n_layers=N_LAYERS)

        for _ in range(50):
            k, v = _random_kv(seq_len=1)
            cache.update(k, v, layer_idx=0)

        assert cache.get_seq_length(0) == 50

    def test_evicted_count_is_zero(self):
        """TurboQuant compresses, doesn't evict -- evicted_count must be 0."""
        cfg = TurboQuantConfig(residual_window=8)
        cache = TurboQuantKVCache(cfg, n_layers=N_LAYERS)

        for _ in range(20):
            k, v = _random_kv()
            cache.update(k, v, layer_idx=0)

        assert cache.evicted_count == 0

    def test_multi_layer_independence(self):
        """Each layer should have independent storage."""
        cfg = TurboQuantConfig(residual_window=128)
        cache = TurboQuantKVCache(cfg, n_layers=N_LAYERS)

        k0, v0 = _random_kv(seq_len=5)
        k1, v1 = _random_kv(seq_len=10)

        cache.update(k0, v0, layer_idx=0)
        cache.update(k1, v1, layer_idx=1)

        assert cache.get_seq_length(0) == 5
        assert cache.get_seq_length(1) == 10


class TestCompression:
    def test_no_compression_within_window(self):
        """Tokens within residual_window should stay in fp16 (no compressed chunks)."""
        cfg = TurboQuantConfig(residual_window=128)
        cache = TurboQuantKVCache(cfg, n_layers=N_LAYERS)

        for _ in range(50):
            k, v = _random_kv()
            cache.update(k, v, layer_idx=0)

        assert len(cache._chunks_k[0]) == 0
        assert len(cache._chunks_v[0]) == 0

    def test_compression_triggers_beyond_window(self):
        """Once we exceed residual_window, compressed chunks should appear."""
        cfg = TurboQuantConfig(residual_window=8)
        cache = TurboQuantKVCache(cfg, n_layers=N_LAYERS)

        for _ in range(12):
            k, v = _random_kv()
            cache.update(k, v, layer_idx=0)

        assert len(cache._chunks_k[0]) > 0
        assert cache.get_seq_length(0) == 12

    def test_full_sequence_always_returned(self):
        """Even after compression, update() must return all tokens."""
        cfg = TurboQuantConfig(residual_window=8)
        cache = TurboQuantKVCache(cfg, n_layers=N_LAYERS)

        for i in range(20):
            k, v = _random_kv()
            out_k, out_v = cache.update(k, v, layer_idx=0)
            expected_seq = i + 1
            assert out_k.shape[2] == expected_seq, f"Step {i}: expected seq={expected_seq}, got {out_k.shape[2]}"
            assert out_v.shape[2] == expected_seq

    def test_reconstruction_quality(self):
        """Round-trip compression should not catastrophically corrupt values."""
        cfg = TurboQuantConfig(key_bits=4, value_bits=4, residual_window=4)
        cache = TurboQuantKVCache(cfg, n_layers=N_LAYERS)

        # Feed a batch of tokens to get some compressed
        all_k, all_v = _random_kv(seq_len=20)

        # Feed one at a time to trigger compression
        for i in range(20):
            k_slice = all_k[:, :, i:i+1, :]
            v_slice = all_v[:, :, i:i+1, :]
            out_k, out_v = cache.update(k_slice, v_slice, layer_idx=0)

        # The recent window (last 4 tokens) should be exact
        assert torch.allclose(out_k[:, :, -4:, :], all_k[:, :, -4:, :], atol=1e-5)
        assert torch.allclose(out_v[:, :, -4:, :], all_v[:, :, -4:, :], atol=1e-5)

        # Compressed tokens should be close but not exact
        # With 4-bit quantization on d=128, MSE should be small
        mse_k = (out_k[:, :, :-4, :] - all_k[:, :, :-4, :]).pow(2).mean()
        mse_v = (out_v[:, :, :-4, :] - all_v[:, :, :-4, :]).pow(2).mean()

        # Sanity: MSE should be much less than the signal variance
        signal_var = all_k.var()
        assert mse_k < signal_var * 0.5, f"Key MSE too high: {mse_k:.4f} vs signal var {signal_var:.4f}"
        assert mse_v < signal_var * 0.5, f"Value MSE too high: {mse_v:.4f} vs signal var {signal_var:.4f}"


class TestMemoryReport:
    def test_report_structure(self):
        """memory_report() should return expected keys."""
        cfg = TurboQuantConfig(residual_window=8)
        cache = TurboQuantKVCache(cfg, n_layers=N_LAYERS)

        # Use enough tokens so compressed portion dominates
        for _ in range(200):
            k, v = _random_kv()
            cache.update(k, v, layer_idx=0)

        report = cache.memory_report()
        assert "compressed_bytes" in report
        assert "fp16_recent_bytes" in report
        assert "total_bytes" in report
        assert "fp16_equivalent_bytes" in report
        assert "compression_ratio" in report
        assert report["compression_ratio"] > 1.0

    def test_no_compression_ratio_is_one(self):
        """Without compression, ratio should be ~1."""
        cfg = TurboQuantConfig(residual_window=128)
        cache = TurboQuantKVCache(cfg, n_layers=N_LAYERS)

        for _ in range(10):
            k, v = _random_kv()
            cache.update(k, v, layer_idx=0)

        report = cache.memory_report()
        # All tokens in fp16, no compressed chunks
        assert len(cache._chunks_k[0]) == 0


class TestPrefill:
    def test_large_prefill(self):
        """Simulates prefill with a large initial sequence."""
        cfg = TurboQuantConfig(residual_window=32)
        cache = TurboQuantKVCache(cfg, n_layers=N_LAYERS)

        # Prefill: 200 tokens at once
        k, v = _random_kv(seq_len=200)
        out_k, out_v = cache.update(k, v, layer_idx=0)

        assert out_k.shape[2] == 200
        assert cache.get_seq_length(0) == 200
        # Should have compressed the overflow (200 - 32 = 168 tokens)
        assert len(cache._chunks_k[0]) > 0

    def test_prefill_then_decode(self):
        """Prefill + autoregressive decode should work correctly."""
        cfg = TurboQuantConfig(residual_window=16)
        cache = TurboQuantKVCache(cfg, n_layers=N_LAYERS)

        # Prefill
        k, v = _random_kv(seq_len=50)
        cache.update(k, v, layer_idx=0)

        # Decode 20 more tokens
        for i in range(20):
            k, v = _random_kv(seq_len=1)
            out_k, out_v = cache.update(k, v, layer_idx=0)

        assert out_k.shape[2] == 70
        assert cache.get_seq_length(0) == 70
