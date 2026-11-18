"""CPU tests for the shared fitted-levels module."""

from __future__ import annotations

import importlib.util
import os

import pytest

torch = pytest.importorskip("torch")

from turboquant.fitted_levels import assign, fit_levels, meta_scalars, reconstruct  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _core():
    path = os.path.join(_ROOT, "models/Qwen3-TTS/benchmarks/keygauss_core.py")
    spec = importlib.util.spec_from_file_location("keygauss_core", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_minmax_levels_hit_endpoints_and_are_even():
    z = torch.rand(3, 20, 8) * 4 - 2
    lv = fit_levels(z, 2, "minmax")
    assert torch.allclose(lv[:, 0], z.amin(1)) and torch.allclose(lv[:, -1], z.amax(1))
    assert torch.allclose(lv[:, 1] - lv[:, 0], lv[:, 2] - lv[:, 1])


def test_kmeans_is_deterministic_and_never_worse_than_minmax():
    z = 0.3 * torch.randn(4, 53, 16, generator=torch.Generator().manual_seed(0)) + 1.5
    a, b = fit_levels(z, 2, "kmeans"), fit_levels(z, 2, "kmeans")
    assert torch.equal(a, b)

    def err(lv):
        return ((reconstruct(assign(z, lv), lv) - z) ** 2).mean()

    assert err(a) <= err(fit_levels(z, 2, "minmax"))


def test_more_levels_than_samples_is_lossless_and_finite():
    z = torch.randn(2, 5, 4)
    for mode in ("kmeans", "minmax"):
        lv = fit_levels(z, 3, mode)  # 8 levels, 5 samples
        assert torch.isfinite(lv).all()
    lv = fit_levels(z, 3, "kmeans")
    assert torch.allclose(reconstruct(assign(z, lv), lv), z, atol=1e-6)


def test_constant_coordinate_has_no_nan():
    z = torch.ones(1, 10, 3)
    for mode in ("kmeans", "minmax"):
        lv = fit_levels(z, 2, mode)
        assert torch.isfinite(lv).all()
        assert torch.allclose(reconstruct(assign(z, lv), lv), z)


def test_meta_scalars():
    assert meta_scalars(2, "kmeans") == 4 and meta_scalars(3, "minmax") == 2


def test_keygauss_quant_kmeans_still_passes_segment_tests():
    c = _core()
    z = torch.randn(40, 6)
    assert c.quant_kmeans(z, 2).shape == z.shape
