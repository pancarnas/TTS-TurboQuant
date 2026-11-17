"""CPU tests for models/Qwen3-TTS/benchmarks/keygauss_core.py (no model needed)."""
from __future__ import annotations

import importlib.util
import math
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


def test_window_stats_match_gaussian_floor():
    c = _core()
    st = c.window_stats(_gauss(), g=32)
    # biased variance of 32 N(0,1) samples retains 31/32; W1/KS floors ~0.2 / ~0.15
    assert abs(st["var_w"] - 31 / 32) < 0.03
    assert 0.15 < st["w1_w"] < 0.27
    assert 0.11 < st["ks_w"] < 0.19
    assert st["n_win"] == 20 * 64


def test_window_stats_detect_offset_and_narrow_spread():
    c = _core()
    z = 0.3 * _gauss() + 1.5  # narrow and shifted, like a key coordinate under a fixed rotation
    st = c.window_stats(z, g=32)
    gl = c.global_stats(z, g=32)
    assert st["var_w"] < 0.12
    assert st["w1_w"] > 1.0 and st["ks_w"] > 0.5
    assert abs(gl["offset"] - 2.25) < 0.1  # mean^2 = 1.5^2
    assert abs(gl["drift"]) < 0.05  # no temporal structure


def test_drift_is_high_when_windows_sit_at_different_levels():
    c = _core()
    S, D = 640, 64
    levels = torch.repeat_interleave(torch.randn(S // 32, D, generator=torch.Generator().manual_seed(1)), 32, dim=0)
    z = levels * 3 + 0.2 * _gauss(S, D)
    gl = c.global_stats(z, g=32)
    assert gl["drift"] > 0.9


def test_gaussian_reference_and_rotate_unit_scale():
    c = _core()
    ref = c.gaussian_reference(640, 64, g=32, seed=3)
    assert abs(ref["var_w"] - 31 / 32) < 0.03
    Pi = c.layer_rotation(64, layer_idx=2)
    z = c.rotate_unit(torch.randn(100, 64), Pi)
    # unit vectors scaled by sqrt(D): every row has squared norm D
    assert torch.allclose((z ** 2).sum(-1), torch.full((100,), 64.0), atol=1e-3)


def test_quant_errors_rank_the_fixes_as_expected():
    c = _core()
    D = 64
    cb = c.codebook(D, 2)
    # Gaussian coordinates: fixed Lloyd-Max is near its design distortion, affine ~ same
    z = _gauss(1024, D)
    mu, sigma = z.mean(0), z.std(0)
    e = c.quant_errors(z, 2, cb, g=32, mu=mu, sigma=sigma)
    assert 0.08 < e["relmse_fixed_b2"] < 0.16
    assert abs(e["relmse_affine_b2"] - e["relmse_fixed_b2"]) < 0.02
    # shifted/narrow coordinates: fixed codebook collapses, affine and g32 recover
    z2 = 0.2 * _gauss(1024, D, seed=5) + 2.0
    mu2, sigma2 = z2.mean(0), z2.std(0)
    e2 = c.quant_errors(z2, 2, cb, g=32, mu=mu2, sigma=sigma2)
    assert e2["relmse_fixed_b2"] > 0.05
    assert e2["relmse_affine_b2"] < 0.2 * e2["relmse_fixed_b2"]
    assert e2["relmse_g32_b2"] < 0.2 * e2["relmse_fixed_b2"]
    # drifting levels: g32 beats static affine
    levels = torch.repeat_interleave(torch.randn(32, D, generator=torch.Generator().manual_seed(2)), 32, dim=0)
    z3 = levels * 2 + 0.2 * _gauss(1024, D, seed=6)
    e3 = c.quant_errors(z3, 2, cb, g=32, mu=z3.mean(0), sigma=z3.std(0))
    assert e3["relmse_g32_b2"] < 0.5 * e3["relmse_affine_b2"]


def test_affine_calib_running_stats():
    c = _core()
    cal = c.AffineCalib()
    z = _gauss(300, 8) * 2 + 1
    cal.add(("k",), z[:100])
    cal.add(("k",), z[100:])
    mu, sigma = cal.get(("k",))
    assert torch.allclose(mu, z.mean(0), atol=1e-4)
    assert torch.allclose(sigma, z.std(0, unbiased=False), atol=1e-3)
    assert cal.get(("missing",)) == (None, None)


def test_alignment_stats_detect_low_rank_queries():
    c = _core()
    D = 32
    g = torch.Generator().manual_seed(0)
    iso_q, iso_k = torch.randn(500, D, generator=g), torch.randn(500, D, generator=g)
    a = c.alignment_stats(iso_q, iso_k, topk=4)
    assert a["q_erank"] > 25 and abs(a["k_share_topq"] - 4 / D) < 0.06
    # queries confined to 2 directions and keys varying mostly along them
    basis = torch.linalg.qr(torch.randn(D, D, generator=g))[0][:, :2]
    low_q = torch.randn(500, 2, generator=g) @ basis.T + 0.01 * torch.randn(500, D, generator=g)
    k = torch.randn(500, 2, generator=g) @ basis.T * 3 + torch.randn(500, D, generator=g)
    b = c.alignment_stats(low_q, k, topk=4)
    assert b["q_erank"] < 4
    assert b["k_share_topq"] > 0.3


def test_attention_mass_sums_to_one_and_respects_prefix():
    c = _core()
    Hq, Hkv, S, D = 4, 2, 50, 16
    q = torch.randn(Hq, 1, D)
    k = torch.randn(Hkv, S, D)
    m_all, m_prompt = c.attention_mass(q, k, groups=2, scaling=1 / math.sqrt(D), first=S, prompt_len=10)
    assert abs(m_all - 1.0) < 1e-5 and 0 < m_prompt < 1
    # a key with a huge logit at position 0 pulls all mass to the first token
    k2 = k.clone()
    k2[:, 0] = 50 * q[0, 0]
    m_first, _ = c.attention_mass(q[:1].repeat(Hq, 1, 1), k2, groups=2, scaling=1 / math.sqrt(D), first=1, prompt_len=10)
    assert m_first > 0.99


def test_layer_rows_shapes_and_calibration_flow():
    c = _core()
    Hkv, S, D = 2, 200, 32
    keys, values = torch.randn(Hkv, S, D), torch.randn(Hkv, S, D)
    Pi = c.layer_rotation(D, 0)
    cbs = {2: c.codebook(D, 2)}
    cal = c.AffineCalib()
    rows = c.layer_rows(0, keys, values, keys * 0.9, prompt_len=50, Pi=Pi, bits_list=[2], centroids=cbs,
                        g=32, calib=cal, evaluate_affine=False, ident=dict(group="g", idx=0))
    # 2 heads x (3 tensors x 3 segments + gaussian)
    assert len(rows) == 2 * (3 * 3 + 1)
    assert all(math.isnan(r["relmse_affine_b2"]) for r in rows if r.get("tensor") == "key" and r["segment"] == "all")
    rows2 = c.layer_rows(0, keys, values, None, prompt_len=50, Pi=Pi, bits_list=[2], centroids=cbs,
                         g=32, calib=cal, evaluate_affine=True, ident=dict(group="g", idx=1))
    ka = [r for r in rows2 if r.get("tensor") == "key" and r["segment"] == "all"]
    assert len(ka) == 2 and all(not math.isnan(r["relmse_affine_b2"]) for r in ka)
    assert {r["tensor"] for r in rows2} == {"key", "value", "gaussian"}


class _FakeAttn:
    def __init__(self, layer_idx, groups=2, name="Qwen3TTSTalkerAttention"):
        self.layer_idx = layer_idx
        self.num_key_value_groups = groups
        self.rope_scaling = {}
        self._tq_rope = None
        self.__class__ = type(name, (_FakeAttn,), {})


def _rope_fn(k, cos, sin, mrope_section=None, interleaved=False, inverse=False):
    # a stand-in for rope_keys: scale by cos (tables are (1, S, D)); inverse divides
    c = cos.to(k.dtype)
    return k / c if inverse else k * c


def test_recorder_collects_cache_tables_and_queries_then_finalizes():
    c = _core()
    Hq, Hkv, D, P, steps = 4, 2, 32, 40, 70
    rec = c.KeyStatsRecorder("Qwen3TTSTalkerAttention", [2], 32, 16, 42, calib_sentences=1, rope_fn=_rope_fn)
    rec.active = True
    g = torch.Generator().manual_seed(0)
    talker = [_FakeAttn(0), _FakeAttn(1)]
    codepred = _FakeAttn(0, name="Qwen3TTSAttention")
    K = [torch.randn(1, Hkv, P + steps, D, generator=g) for _ in talker]
    V = [torch.randn(1, Hkv, P + steps, D, generator=g) for _ in talker]
    cos_all = torch.full((1, P + steps, D), 2.0)
    for S in [P] + list(range(P + 1, P + steps + 1)):
        new = P if S == P else 1
        q = torch.randn(1, Hq, new, D, generator=g)
        for li, m in enumerate(talker):
            m._tq_rope = (cos_all[:, S - new:S], torch.zeros(1, new, D))
            rec.record(m, q, K[li][:, :, :S], V[li][:, :, :S], None, 1 / math.sqrt(D))
        codepred._tq_rope = (torch.ones(1, 1, D), torch.zeros(1, 1, D))
        rec.record(codepred, q[:, :, :1], torch.randn(1, Hkv, 5, D), torch.randn(1, Hkv, 5, D), None, 0.1)
    assert rec.prompt_len == P and rec.rope[0].shape[-2] == P + steps
    assert set(rec.kv) == {0, 1} and rec.seen_classes["Qwen3TTSAttention"] == steps + 1
    rows, ls = rec.finalize(dict(group="g", idx=0))
    assert len(ls) == 2 and all(l["pre_rope_ok"] for l in ls) and ls[0]["n_tok"] == P + steps
    assert 0 < ls[0]["attn_mass_first"] <= 1 and 0 < ls[0]["attn_mass_prompt"] <= 1
    # keys_pre = keys / 2 has the same direction -> identical window stats as 'key'
    df = __import__("pandas").DataFrame(rows)
    a = df[(df.tensor == "key") & (df.segment == "all") & (df.layer == 0) & (df.kv_head == 0)].iloc[0]
    b = df[(df.tensor == "key_pre") & (df.segment == "all") & (df.layer == 0) & (df.kv_head == 0)].iloc[0]
    assert abs(a.var_w - b.var_w) < 1e-5
    # first sentence calibrates only: affine is NaN; second sentence evaluates it
    assert math.isnan(a.relmse_affine_b2)
    assert rec.sentence_no == 1 and not rec.kv
    rec.record(talker[0], torch.randn(1, Hq, P, D), K[0][:, :, :P], V[0][:, :, :P], None, 0.1)
    talker[0]._tq_rope = (cos_all[:, :P], torch.zeros(1, P, D))
    rows2, ls2 = rec.finalize(dict(group="g", idx=1))
    df2 = __import__("pandas").DataFrame(rows2)
    a2 = df2[(df2.tensor == "key") & (df2.segment == "all")].iloc[0]
    assert not math.isnan(a2.relmse_affine_b2)
    md = c.summarize(__import__("pandas").concat([df, df2]), __import__("pandas").DataFrame(ls + ls2), 32)
    assert "| Gaussian |" in md and "| Key (post-RoPE) |" in md and "## Per layer" in md


def test_make_patch_swallows_recorder_errors():
    c = _core()
    rec = c.KeyStatsRecorder("Qwen3TTSTalkerAttention", [2], 32, 16, 42, 0)
    rec.active = True
    calls = []
    def original(module, q, k, v, m, s, dropout=0.0, **kw):
        calls.append(1)
        return "out", None
    patched = c.make_patch(original, rec)
    out, _ = patched(_FakeAttn(0), torch.zeros(1, 2, 3, 4), None, None, None, 1.0)
    assert out == "out" and calls == [1] and rec.errors == 1 and "Traceback" in rec.first_error
