"""Prefix codebook (docs/superpowers plan 2026-11-18): fitted-levels stage."""

import pytest
import torch

from turboquant.compressors_v3 import MSECompressor, compressed_bytes_per_token


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
