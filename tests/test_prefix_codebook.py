"""Prefix codebook (docs/superpowers plan 2026-11-18): fitted-levels stage."""

import pytest
import torch

from turboquant.config import TurboQuantConfig
from turboquant.compressors_v3 import (
    MSECompressor,
    TurboQuantV3,
    compressed_bytes_per_token,
)


def test_bytes_per_token_adds_amortized_metadata_and_keeps_old_signature():
    assert compressed_bytes_per_token(128, 2) == compressed_bytes_per_token(128, 2, 0)
    assert compressed_bytes_per_token(
        128, 2, 0, meta_scalars=4, block=53
    ) == pytest.approx(compressed_bytes_per_token(128, 2) + 4 * 128 * 2 / 53)


def test_bytes_per_token_meta_without_block_raises():
    with pytest.raises(ValueError):
        compressed_bytes_per_token(128, 2, 0, meta_scalars=4)


def test_levels_mode_with_group_raises():
    with pytest.raises(ValueError):
        MSECompressor(16, 2, seed=1, group=32, levels_mode="kmeans")


@pytest.mark.parametrize("mode", ["kmeans", "minmax"])
def test_levels_compressor_round_trip_is_lossy_but_close(mode):
    c = MSECompressor(32, 3, seed=1, levels_mode=mode)
    x = torch.randn(1, 2, 53, 32)
    out = c.compress(x)
    key = "levels" if mode == "kmeans" else "lo"
    assert out[key].dtype == torch.float16
    y = c.decompress(out)
    assert y.shape == x.shape
    cos = torch.nn.functional.cosine_similarity(x.flatten(2), y.flatten(2), dim=-1)
    assert cos.mean() > 0.98


def test_levels_compressor_prefix_longer_than_levels_is_near_exact():
    c = MSECompressor(16, 8, seed=1, levels_mode="kmeans")  # 256 levels on 40 tokens
    x = torch.randn(1, 1, 40, 16)
    y = c.decompress(c.compress(x))
    assert torch.allclose(x, y, atol=2e-3)  # fp16 norm + fp16 levels only


def test_levels_mode_off_is_bitwise_old_path():
    a, b = MSECompressor(16, 2, seed=1), MSECompressor(16, 2, seed=1, levels_mode="")
    x = torch.randn(1, 1, 30, 16)
    assert torch.equal(a.decompress(a.compress(x)), b.decompress(b.compress(x)))


def test_memory_bytes_counts_levels_over_the_block():
    c = MSECompressor(32, 2, seed=1, levels_mode="kmeans")
    assert c.meta_scalars == 4
    expected = round(2 * 53 * compressed_bytes_per_token(32, 2, 0, 4, 53))
    assert c.memory_bytes(1, 2, 53)["compressed_bytes"] == expected


def test_v3_prefix_zero_is_identical_to_old_output():
    c = TurboQuantV3(
        16,
        key_bits=2,
        value_bits=2,
        residual_window=0,
        protected_layers=0,
        key_group=16,
    )
    k = torch.randn(1, 2, 64, 16)
    v = torch.randn(1, 2, 64, 16)
    a = c.decompress_kv(*c.compress_kv(k, v))
    b = c.decompress_kv(*c.compress_kv(k, v, prefix=0))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_v3_prefix_compressed_with_fitted_levels_rest_unchanged():
    base = TurboQuantV3(16, 2, 2, residual_window=0, protected_layers=0, key_group=16)
    pq = TurboQuantV3(
        16,
        2,
        2,
        residual_window=0,
        protected_layers=0,
        key_group=16,
        prefix_codebook="kmeans",
        prefix_bits=3,
    )
    assert pq.prefix_bits_effective == 3 and pq.prefix_key_compressor.bits == 3
    k = torch.randn(1, 2, 70, 16)
    v = torch.randn(1, 2, 70, 16)
    ck, cv = pq.compress_kv(k, v, prefix=22)
    assert ck["prefix"]["n"] == 22 and "levels" in ck["prefix"]["compressed"]
    rk, rv = pq.decompress_kv(ck, cv)
    assert rk.shape == k.shape
    fixed = base.decompress_kv(*base.compress_kv(k[:, :, :22], v[:, :, :22]))[0]

    def err(a, b):
        return ((a - b) ** 2).sum() / (b**2).sum()

    assert err(rk[:, :, :22], k[:, :, :22]) < 0.5 * err(fixed, k[:, :, :22])
    rest = base.decompress_kv(*base.compress_kv(k[:, :, 22:], v[:, :, 22:]))[0]
    assert torch.equal(rk[:, :, 22:], rest)


def test_v3_prefix_only_empty_remainder():
    pq = TurboQuantV3(
        16, 2, 2, protected_layers=0, prefix_codebook="minmax", prefix_bits=3
    )
    k = torch.randn(1, 2, 22, 16)
    v = torch.randn(1, 2, 22, 16)
    ck, cv = pq.compress_kv(k, v, prefix=22)
    rk, rv = pq.decompress_kv(ck, cv)
    assert rk.shape == k.shape and rv.shape == v.shape


def test_v3_prefix_bits_zero_follows_key_bits():
    c = TurboQuantV3(16, 8, 8, protected_layers=0, prefix_codebook="minmax")
    assert c.prefix_bits_effective == 8


def test_v3_rejects_bad_args():
    with pytest.raises(ValueError):
        TurboQuantV3(16, 2, 2, prefix_codebook="median")
    with pytest.raises(ValueError):
        TurboQuantV3(16, 2, 2, prefix_codebook="kmeans", prefix_bits=-1)
    with pytest.raises(ValueError):
        TurboQuantV3(16, 2, 2).compress_kv(
            torch.randn(1, 1, 4, 16), torch.randn(1, 1, 4, 16), prefix=2
        )


def test_config_defaults_off_and_validation():
    c = TurboQuantConfig()
    assert c.prefix_codebook == "" and c.prefix_bits == 0
    TurboQuantConfig(protect_prefix=-1, prefix_codebook="kmeans", prefix_bits=2)
    with pytest.raises(ValueError):
        TurboQuantConfig(protect_prefix=0, prefix_codebook="kmeans")
    with pytest.raises(ValueError):
        TurboQuantConfig(protect_prefix=-1, prefix_codebook="other")
