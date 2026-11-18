"""CPU tests for the per-segment quantization-shift stats in keygauss_core."""

from __future__ import annotations

import importlib.util
import os

import pytest

torch = pytest.importorskip("torch")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _core():
    path = os.path.join(_ROOT, "models/Qwen3-TTS/benchmarks/keygauss_core.py")
    spec = importlib.util.spec_from_file_location("keygauss_core", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _gauss(S=640, D=64, seed=0):
    return torch.randn(S, D, generator=torch.Generator().manual_seed(seed))


def test_kmeans_beats_fixed_on_shifted_narrow_keys():
    c = _core()
    z = 0.3 * _gauss() + 1.5  # prompt-like: narrow spread, large offset
    cb = c.codebook(64, 2)
    fixed = c._relmse(z, c.quant_fixed(z, cb))
    km = c._relmse(z, c.quant_kmeans(z, 2))
    assert km < 0.2 * fixed


def test_fixed_codebook_shifts_offset_keys_but_minmax_does_not():
    c = _core()
    z = (
        0.3 * _gauss() + 2.5
    )  # beyond the outer Lloyd-Max level (1.51): every value saturates
    out = c.quant_errors(z, 2, c.codebook(64, 2), g=32)
    assert out["shift_fixed_b2"] > 0.05  # saturation pulls the mean toward the codebook
    assert out["shift_g32_b2"] < 0.005  # min-max rounding is unbiased
    assert out["shift_kmeans_b2"] < 0.005


def test_relmsec_flags_loss_of_token_identity():
    c = _core()
    z = 0.3 * _gauss() + 1.5
    out = c.quant_errors(z, 2, c.codebook(64, 2), g=32)
    # the fixed codebook maps almost every value of a narrow coordinate to one level
    assert out["relmsec_fixed_b2"] > 0.5
    assert out["relmsec_fixed_b2"] > 3 * out["relmse_fixed_b2"]
    assert out["relmsec_g32_b2"] < out["relmsec_fixed_b2"]
    assert set(k.split("_")[1] for k in out if k.startswith("relmse_")) >= {
        "fixed",
        "g32",
        "gseg",
        "kmeans",
    }


def test_layer_rows_quantize_every_key_segment():
    c = _core()
    S, D, P = 96, 32, 40
    keys, values = _gauss(S, D, 1).unsqueeze(0), _gauss(S, D, 2).unsqueeze(0)
    Pi = c.layer_rotation(D, 0)
    rows = c.layer_rows(
        0, keys, values, None, P, Pi, [2], {2: c.codebook(D, 2)}, 32, None, False, {}
    )
    segs = {
        r["segment"] for r in rows if r["tensor"] == "key" and "relmse_kmeans_b2" in r
    }
    assert segs == {"all", "prompt", "gen"}
