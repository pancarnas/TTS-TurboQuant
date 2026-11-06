"""Key-fix experiments (docs/keyfix_experiment.md): group key scaling (E1) and
pre-RoPE key quantization (E2). Model-free, CPU only.

Covers the compressor, the shared byte accounting, both caches' span rounding,
the RoPE undo/redo helper, the Qwen cache's pre-RoPE write-back path, the
divergence recorder in pre-RoPE mode, and the wav-name parsers.
"""

from __future__ import annotations

import importlib.util
import os
import types

import pytest

torch = pytest.importorskip("torch")

from turboquant.compressors_v3 import (  # noqa: E402
    MSECompressor,
    TurboQuantV3,
    compressed_bytes_per_token,
)
from turboquant.config import TurboQuantConfig  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _qwen_cache_mod():
    pytest.importorskip("transformers")
    return _load(
        os.path.join(_ROOT, "models/Qwen3-TTS/qwen_tts/core/models/turboquant_kv_cache.py"),
        "tq_kv_cache_keyfix",
    )


def _vallex_cache_mod():
    return _load(os.path.join(_ROOT, "models/VALL-E-X/turboquant_cache.py"), "tq_vallex_cache_keyfix")


# ---------------------------------------------------------------------------
# E1: group key scaling in MSECompressor
# ---------------------------------------------------------------------------


def _narrow_band_keys(B=1, H=2, S=64, D=64, seed=0):
    """Keys whose rotated coordinates drift slowly over time (the regime the
    paper's Table 6 describes): a smooth random walk plus small noise."""
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(B, H, 1, D, generator=g)
    drift = torch.cumsum(0.05 * torch.randn(B, H, S, D, generator=g), dim=2)
    return base + drift + 0.02 * torch.randn(B, H, S, D, generator=g)


def _cos(a, b):
    return torch.nn.functional.cosine_similarity(
        a.reshape(-1, a.shape[-1]).float(), b.reshape(-1, b.shape[-1]).float(), dim=-1
    ).mean().item()


def test_group_compressor_round_trip_shapes_and_keys():
    x = _narrow_band_keys()
    c = MSECompressor(64, 2, seed=1, group=32)
    blob = c.compress(x)
    assert blob["gmin"].shape == (1, 2, 2, 64) and blob["gscale"].shape == (1, 2, 2, 64)
    assert blob["gmin"].dtype == torch.float16
    y = c.decompress(blob)
    assert y.shape == x.shape
    assert _cos(x, y) > 0.9


def test_group_scaling_beats_fixed_codebook_on_drifting_keys():
    """The paper's central claim, in our quantizer: for slowly drifting keys a
    per-group min-max range at 2 bits reconstructs better than the fixed
    Gaussian codebook at 2 bits."""
    x = _narrow_band_keys(S=128)
    fixed = MSECompressor(64, 2, seed=1, group=0)
    grouped = MSECompressor(64, 2, seed=1, group=32)
    e_fixed = (fixed.decompress(fixed.compress(x)) - x).pow(2).mean().item()
    e_group = (grouped.decompress(grouped.compress(x)) - x).pow(2).mean().item()
    assert e_group < e_fixed


def test_group_zero_is_the_fixed_codebook_path():
    x = torch.randn(1, 2, 40, 64)
    c0 = MSECompressor(64, 3, seed=5, group=0)
    blob = c0.compress(x)
    assert "gmin" not in blob
    # same rotation + Lloyd-Max centroids as an explicitly default-constructed compressor
    c_def = MSECompressor(64, 3, seed=5)
    assert torch.equal(c0.decompress(blob), c_def.decompress(c_def.compress(x)))


def test_partial_trailing_group_matches_separate_compression():
    """Padding the last group by repeating its final token must not change its
    min/max: compressing [0:50) at once == compressing [0:32) and [32:50)."""
    x = _narrow_band_keys(S=50)
    c = MSECompressor(64, 2, seed=3, group=32)
    whole = c.decompress(c.compress(x))
    a = c.decompress(c.compress(x[:, :, :32]))
    b = c.decompress(c.compress(x[:, :, 32:]))
    assert torch.allclose(whole, torch.cat([a, b], dim=2), atol=1e-6)


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_group_scaling_works_at_every_bit_width(bits):
    x = _narrow_band_keys(S=32)
    c = MSECompressor(64, bits, seed=2, group=32)
    y = c.decompress(c.compress(x))
    assert _cos(x, y) > {2: 0.9, 3: 0.97, 4: 0.99, 8: 0.9999}[bits]


def test_compressed_bytes_per_token_counts_group_metadata_honestly():
    # D=64: 2-bit idx = 16 B + 2 B norm = 18 B; group=32 adds 2*64*2/32 = 8 B (= 1 bit/elem)
    assert compressed_bytes_per_token(64, 2) == 18
    assert compressed_bytes_per_token(64, 2, 32) == 26
    assert compressed_bytes_per_token(64, 2, 64) == 22
    # nibble-granularity for 3-bit, as before
    assert compressed_bytes_per_token(64, 3) == compressed_bytes_per_token(64, 4) == 34
    m = MSECompressor(64, 2, seed=0, group=32).memory_bytes(1, 1, 32)
    assert m["compressed_bytes"] == 32 * 26


def test_turboquantv3_threads_key_group_to_keys_only():
    tq = TurboQuantV3(head_dim=64, key_bits=2, value_bits=2, layer_idx=0, n_layers=12,
                      protected_layers=0, key_group=32, residual_window=0)
    assert tq.key_compressor.group == 32 and tq.val_compressor.group == 0
    k = _narrow_band_keys(S=64)
    v = torch.randn(1, 2, 64, 64)
    ck, cv = tq.compress_kv(k, v)
    assert "gmin" in ck["compressed"] and "gmin" not in cv["compressed"]
    rk, rv = tq.decompress_kv(ck, cv)
    assert rk.shape == k.shape and rv.shape == v.shape
    assert tq.memory_bytes(1, 2, 64)["key_group"] == 32


# ---------------------------------------------------------------------------
# E1: both caches round compressed spans to whole groups
# ---------------------------------------------------------------------------


def _cfg(**kw):
    base = dict(key_bits=2, value_bits=2, residual_window=0, protected_layers=0,
                protected_bits=8, track_only=False)
    base.update(kw)
    return TurboQuantConfig(**base)


def _feed_qwen(cache, steps, step_len, D=64, H=2, seed=0, with_rope=False, rope_scaling=None):
    g = torch.Generator().manual_seed(seed)
    ks, full = [], None
    for i in range(steps):
        k = torch.randn(1, H, step_len, D, generator=g)
        v = torch.randn(1, H, step_len, D, generator=g)
        ks.append(k)
        kw = None
        if with_rope:
            s = i * step_len
            kw = _rope_kwargs_for_positions(s, step_len, D, rope_scaling)
        full, _ = cache.update(k, v, layer_idx=0, cache_kwargs=kw)
    return torch.cat(ks, dim=2), full


def test_qwen_cache_rounds_spans_to_whole_groups():
    mod = _qwen_cache_mod()
    cache = mod.TurboQuantKVCache(_cfg(key_group=32), n_layers=1)
    k_in, k_out = _feed_qwen(cache, steps=5, step_len=8)  # 40 tokens
    assert not torch.allclose(k_out[:, :, :32], k_in[:, :, :32], atol=1e-3)  # first group lossy
    assert torch.equal(k_out[:, :, 32:], k_in[:, :, 32:])  # partial second group still exact
    _feed_qwen(cache, steps=3, step_len=8, seed=1)  # +24 -> 64 tokens: second group fills
    assert cache._compressed_upto[0] == 64
    assert cache.memory_report()["compressed_bytes"] > 0


def test_qwen_cache_group_zero_behaviour_unchanged():
    mod = _qwen_cache_mod()
    cache = mod.TurboQuantKVCache(_cfg(residual_window=16), n_layers=1)
    k_in, k_out = _feed_qwen(cache, steps=6, step_len=8)
    assert torch.equal(k_out[:, :, -16:], k_in[:, :, -16:])
    assert not torch.allclose(k_out[:, :, :-16], k_in[:, :, :-16], atol=1e-3)
    assert cache._compressed_upto[0] == 32


def test_vallex_cache_rounds_spans_to_whole_groups():
    mod = _vallex_cache_mod()
    cache = mod.TurboQuantValleCache(_cfg(key_group=32), n_layers=1)
    g = torch.Generator().manual_seed(0)
    ks, full = [], None
    for _ in range(5):  # 40 tokens
        k = torch.randn(1, 2, 8, 64, generator=g)
        v = torch.randn(1, 2, 8, 64, generator=g)
        ks.append(k)
        full, _ = cache.update(0, k, v)
    k_in = torch.cat(ks, dim=2)
    assert not torch.allclose(full[:, :, :32], k_in[:, :, :32], atol=1e-3)
    assert torch.equal(full[:, :, 32:], k_in[:, :, 32:])
    rep = cache.memory_report()
    assert rep["simulated_compressed_bytes"] > 0


# ---------------------------------------------------------------------------
# E2: RoPE undo/redo helper
# ---------------------------------------------------------------------------


def _tables(S, D, lead=(), seed=0):
    """Raw rotary tables shaped like the model's: cat([f, f]) over the last dim."""
    g = torch.Generator().manual_seed(seed)
    theta = torch.rand(*lead, S, D // 2, generator=g) * 6.283
    emb = torch.cat([theta, theta], dim=-1)
    return emb.cos(), emb.sin()


def _rope_kwargs_for_positions(start, n, D, rope_scaling):
    """Deterministic raw tables for positions [start, start+n) — mrope (3,B,S,D)
    when rope_scaling has an mrope_section, else plain (B,S,D)."""
    pos = torch.arange(start, start + n).float()
    inv = 1.0 / (10000 ** (torch.arange(0, D, 2).float() / D))
    freqs = pos[:, None] * inv[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)[None]  # (1, n, D)
    cos, sin = emb.cos(), emb.sin()
    if rope_scaling and rope_scaling.get("mrope_section"):
        cos, sin = cos[None].expand(3, 1, n, D).clone(), sin[None].expand(3, 1, n, D).clone()
    return {"cos": cos, "sin": sin, "cache_position": torch.arange(start, start + n)}


@pytest.mark.parametrize(
    "lead,section,interleaved",
    [((), None, False), ((3,), [16, 24, 24], False), ((3,), [16, 24, 24], True)],
)
def test_rope_keys_inverse_is_exact(lead, section, interleaved):
    mod = _qwen_cache_mod()
    D, S = 128, 20
    k = torch.randn(1, 4, S, D)
    cos, sin = _tables(S, D, lead=lead + (1,))  # (B, S, D) or (3, B, S, D)
    k_rot = mod.rope_keys(k, cos, sin, mrope_section=section, interleaved=interleaved)
    assert not torch.allclose(k_rot, k, atol=1e-3)  # RoPE actually did something
    back = mod.rope_keys(k_rot, cos, sin, mrope_section=section, interleaved=interleaved, inverse=True)
    assert torch.allclose(back, k, atol=1e-5)


def test_rope_keys_rejects_bad_tables():
    mod = _qwen_cache_mod()
    k = torch.randn(1, 2, 4, 16)
    cos, sin = _tables(4, 16, lead=(3, 1))
    with pytest.raises(ValueError):
        mod.rope_keys(k, cos, sin)  # 4-D tables need an mrope_section


# ---------------------------------------------------------------------------
# E2: Qwen cache pre-RoPE write-back
# ---------------------------------------------------------------------------

_MROPE = {"mrope_section": [8, 12, 12], "interleaved": False}  # sums to D/2 for D=64


def test_qwen_cache_pre_rope_needs_tables():
    mod = _qwen_cache_mod()
    cache = mod.TurboQuantKVCache(_cfg(pre_rope_keys=True), n_layers=1, rope_scaling=_MROPE)
    with pytest.raises(RuntimeError):
        cache.update(torch.randn(1, 2, 4, 64), torch.randn(1, 2, 4, 64), layer_idx=0)


def test_qwen_cache_pre_rope_8bit_is_near_identity():
    """With 8-bit keys the quantizer is ~lossless, so any error left in the
    read-back would come from the RoPE undo/redo — it must be ~exact."""
    mod = _qwen_cache_mod()
    cache = mod.TurboQuantKVCache(_cfg(key_bits=8, value_bits=8, pre_rope_keys=True),
                                  n_layers=1, rope_scaling=_MROPE)
    k_in, k_out = _feed_qwen(cache, steps=6, step_len=8, with_rope=True, rope_scaling=_MROPE)
    assert cache._compressed_upto[0] == 48
    assert _cos(k_in, k_out) > 0.9999
    assert cache._rope_len == 48


def test_qwen_cache_pre_rope_changes_2bit_reconstruction():
    """Pre- and post-RoPE 2-bit quantization are different codecs: the read-back
    must differ (otherwise the flag is a no-op) while both stay lossy."""
    mod = _qwen_cache_mod()
    post = mod.TurboQuantKVCache(_cfg(), n_layers=1, rope_scaling=_MROPE)
    pre = mod.TurboQuantKVCache(_cfg(pre_rope_keys=True), n_layers=1, rope_scaling=_MROPE)
    k_in, k_post = _feed_qwen(post, steps=4, step_len=8, with_rope=True, rope_scaling=_MROPE)
    _, k_pre = _feed_qwen(pre, steps=4, step_len=8, with_rope=True, rope_scaling=_MROPE)
    assert not torch.allclose(k_post, k_in, atol=1e-3)
    assert not torch.allclose(k_pre, k_in, atol=1e-3)
    assert not torch.allclose(k_pre, k_post, atol=1e-3)


def test_qwen_cache_pre_rope_with_plain_tables_and_group():
    """Plain 1-D tables (no rope_scaling) + group scaling together."""
    mod = _qwen_cache_mod()
    cache = mod.TurboQuantKVCache(_cfg(key_group=16, pre_rope_keys=True), n_layers=1)
    k_in, k_out = _feed_qwen(cache, steps=5, step_len=8, with_rope=True, rope_scaling=None)
    assert cache._compressed_upto[0] == 32  # 40 tokens -> two whole groups of 16
    assert torch.equal(k_out[:, :, 32:], k_in[:, :, 32:])


# ---------------------------------------------------------------------------
# E2: divergence recorder in pre-RoPE mode (exec-extracted like the other tests)
# ---------------------------------------------------------------------------

_EXP_PATH = os.path.join(_ROOT, "models/Qwen3-TTS/benchmarks/kv_attn_divergence_experiment.py")


def _recorder_ns():
    import traceback

    from tests.test_kv_attn_divergence import _kv_recon_errors, _repeat_kv  # reuse helpers

    src = open(_EXP_PATH, encoding="utf-8").read()
    start, end = src.index("def js_divergence("), src.index("\ndef force_eager(")
    mod = _qwen_cache_mod()
    ns = {
        "torch": torch,
        "traceback": traceback,
        "modeling": types.SimpleNamespace(repeat_kv=_repeat_kv),
        "TurboQuantV3": TurboQuantV3,
        "_kv_recon_errors": _kv_recon_errors,
        "rope_keys": mod.rope_keys,
        "_N_ID": 10,
    }
    exec(compile(src[start:end], _EXP_PATH, "exec"), ns)
    return ns


def _talker_module(groups, S_new, start, D, layer_idx=0):
    kw = _rope_kwargs_for_positions(start, S_new, D, _MROPE)
    return types.SimpleNamespace(
        layer_idx=layer_idx, num_key_value_groups=groups,
        rope_scaling=_MROPE, _tq_rope=(kw["cos"], kw["sin"]),
    )


def test_recorder_pre_rope_accumulates_tables_and_is_exact_at_8bit():
    ns = _recorder_ns()
    D, groups = 64, 2
    rec = ns["DivergenceRecorder"](specs=[(8, 8, 0), (2, 2, 0)], n_layers=4, prot_layers=0,
                                   prot_bits=8, stride=1, pre_rope=True)
    rec.group, rec.idx = "g", 0
    torch.manual_seed(0)
    q = torch.randn(1, 2 * groups, 1, D)
    key = torch.randn(1, 2, 32, D)
    value = torch.randn(1, 2, 32, D)
    # prefill: 32 positions at once
    rec.record(_talker_module(groups, 32, 0, D), q, key, value, None, D**-0.5)
    # decode: one new position
    key2 = torch.cat([key, torch.randn(1, 2, 1, D)], dim=2)
    value2 = torch.cat([value, torch.randn(1, 2, 1, D)], dim=2)
    rec.record(_talker_module(groups, 1, 32, D), q, key2, value2, None, D**-0.5)
    assert rec.errors == 0, rec.first_error
    assert len(rec.rows) == 4
    rows = [dict(zip(_cols(), r)) for r in rec.rows]
    for d in rows:
        assert d["pre_rope"] == 1 and d["key_group"] == 0
    r8 = [d for d in rows if d["key_bits"] == 8]
    r2 = [d for d in rows if d["key_bits"] == 2]
    assert all(d["cos_k"] > 0.9999 and d["attn_js"] < 1e-5 for d in r8)
    assert all(d["cos_k"] < 0.999 for d in r2)
    assert rec._rope_buf["SimpleNamespace"][0].shape[-2] == 33


def test_recorder_pre_rope_every_layer_sees_the_same_step():
    """Regression: the model has 28 talker layers and each calls record() with
    the SAME step's 1-position tables. The first smoke run raised 'rotary
    tables cover 1 of N positions' on every layer but the first of every
    decode step (46k skipped measurements)."""
    ns = _recorder_ns()
    D, groups, L = 64, 2, 3
    rec = ns["DivergenceRecorder"](specs=[(8, 8, 0)], n_layers=4, prot_layers=0,
                                   prot_bits=8, stride=1, pre_rope=True)
    rec.group, rec.idx = "g", 0
    torch.manual_seed(1)
    q = torch.randn(1, 2 * groups, 1, D)
    key = torch.randn(1, 2, 8, D)
    for layer in range(L):  # prefill, all layers
        rec.record(_talker_module(groups, 8, 0, D, layer), q, key, key, None, D**-0.5)
    for step in range(3):  # three decode steps, all layers each
        key = torch.cat([key, torch.randn(1, 2, 1, D)], dim=2)
        pos = key.shape[2]
        for layer in range(L):
            rec.record(_talker_module(groups, 1, pos - 1, D, layer), q, key, key, None, D**-0.5)
    assert rec.errors == 0 and len(rec.rows) == L * 4
    assert rec._rope_buf["SimpleNamespace"][0].shape[-2] == 11
    rows = [dict(zip(_cols(), r)) for r in rec.rows]
    assert all(d["cos_k"] > 0.9999 and d["attn_js"] < 1e-4 for d in rows)
    # a second sequence of the same class (e.g. next sentence) restarts cleanly
    key = torch.randn(1, 2, 5, D)
    for layer in range(L):
        rec.record(_talker_module(groups, 5, 0, D, layer), q, key, key, None, D**-0.5)
    assert rec._rope_buf["SimpleNamespace"][0].shape[-2] == 5
    # a stale one-position table that matches neither extension nor repeat
    bad = _talker_module(groups, 1, 2, D, 0)  # position 2 while buffer holds 5
    with pytest.raises(RuntimeError, match="rotary tables cover"):
        rec.record(bad, q, key, key, None, D**-0.5)


def test_recorder_pre_rope_accumulates_across_strided_steps():
    """--step-stride skips the measurement, not the table bookkeeping."""
    ns = _recorder_ns()
    D, groups = 64, 2
    rec = ns["DivergenceRecorder"](specs=[(8, 8, 0)], n_layers=4, prot_layers=0,
                                   prot_bits=8, stride=4, pre_rope=True)
    rec.group, rec.idx = "g", 0
    q = torch.randn(1, 2 * groups, 1, D)
    key = torch.randn(1, 2, 4, D)
    rec.record(_talker_module(groups, 4, 0, D), q, key, key, None, D**-0.5)
    for _ in range(4):  # positions 5,6,7 are skipped by the stride, 8 is measured
        key = torch.cat([key, torch.randn(1, 2, 1, D)], dim=2)
        rec.record(_talker_module(groups, 1, key.shape[2] - 1, D), q, key, key, None, D**-0.5)
    assert rec.errors == 0
    assert [r[3] for r in rec.rows] == [4, 8]
    assert rec._rope_buf["SimpleNamespace"][0].shape[-2] == 8


def _cols():
    from tests.test_kv_attn_divergence import COLS

    return COLS


def test_recorder_pre_rope_detects_table_gap():
    ns = _recorder_ns()
    D, groups = 64, 2
    rec = ns["DivergenceRecorder"](specs=[(4, 4, 0)], n_layers=4, prot_layers=0,
                                   prot_bits=8, stride=1, pre_rope=True)
    rec.group, rec.idx = "g", 0
    q = torch.randn(1, 4, 1, D)
    key = torch.randn(1, 2, 16, D)
    # tables cover only 4 of 16 positions on a fresh buffer -> protocol error.
    # record() raises; make_patch() is what turns that into a counted error.
    with pytest.raises(RuntimeError, match="rotary tables cover"):
        rec.record(_talker_module(groups, 4, 0, D), q, key, key, None, D**-0.5)
    assert rec.rows == []


def test_recorder_key_group_threads_into_compressors():
    ns = _recorder_ns()
    D, groups = 64, 2
    rec = ns["DivergenceRecorder"](specs=[(2, 2, 0)], n_layers=4, prot_layers=0,
                                   prot_bits=8, stride=1, key_group=32)
    rec.group, rec.idx = "g", 0
    q = torch.randn(1, 4, 1, D)
    key = _narrow_band_keys(S=64)
    rec.record(_talker_module(groups, 64, 0, D), q, key, key, None, D**-0.5)
    assert rec.errors == 0 and len(rec.rows) == 1
    d = dict(zip(_cols(), rec.rows[0]))
    assert d["key_group"] == 32 and d["pre_rope"] == 0
    comp = next(iter(rec._cache.values()))
    assert comp.key_compressor.group == 32


# ---------------------------------------------------------------------------
# driver helpers + wav-name parsers
# ---------------------------------------------------------------------------


def _single(name, extra):
    src = open(_EXP_PATH, encoding="utf-8").read()
    start = src.index(f"def {name}(")
    end = src.index("\ndef ", start + 1)
    ns = dict(extra)
    exec(compile(src[start:end], _EXP_PATH, "exec"), ns)
    return ns[name]


def test_make_config_threads_keyfix_flags():
    make_config = _single("make_config", {"TurboQuantConfig": TurboQuantConfig})
    c = make_config(2, 2, 0, 0, 32, True)
    assert c.key_group == 32 and c.pre_rope_keys is True and c.track_only is False
    d = make_config(2, 2, 0, 0)
    assert d.key_group == 0 and d.pre_rope_keys is False


def test_wav_path_keyfix_segments_and_fp16_shared(tmp_path):
    wav_path = _single("_wav_path", {"os": os})
    p = wav_path(str(tmp_path), "g", 3, 0, 0.9, 2, 2, 0, 0, 32, True)
    assert p.endswith("qwen_g_3_sampling_s0_t0.9_K2_V2_rw=0_pl=0_g=32_pre.wav")
    assert wav_path(str(tmp_path), "g", 3, 0, 0.9, 2, 2, 0, 0, 32, False).endswith("_pl=0_g=32.wav")
    assert wav_path(str(tmp_path), "g", 3, 0, 0.9, 2, 2, 0, 0, 0, True).endswith("_pl=0_pre.wav")
    assert wav_path(str(tmp_path), "g", 3, 0, 0.9, 2, 2, 0, 0).endswith("_pl=0.wav")
    # fp16 never carries the segments
    assert wav_path(str(tmp_path), "g", 3, 0, 0.9, None, None, None, 0, 32, True).endswith("_fp16.wav")


def test_load_done_filters_keyfix_columns(tmp_path):
    pd = pytest.importorskip("pandas")
    load_done = _single("load_done", {"pd": pd, "os": os})
    cols = _cols()
    rows = [
        ("g", 0, 0, 10, 2, 2, 0, 0, 0, 0) + tuple([0.1] * 9),
        ("g", 1, 0, 10, 2, 2, 0, 0, 32, 0) + tuple([0.1] * 9),
        ("g", 2, 0, 10, 2, 2, 0, 0, 0, 1) + tuple([0.1] * 9),
    ]
    p = tmp_path / "d.csv"
    pd.DataFrame(rows, columns=cols).to_csv(p, index=False)
    assert load_done(str(p), 0) == {("g", 0)}
    assert load_done(str(p), 0, 32) == {("g", 1)}
    assert load_done(str(p), 0, 0, True) == {("g", 2)}


def test_score_wav_dir_parses_keyfix_segments():
    from tools.score_wav_dir import parse_wav_name

    e = parse_wav_name("qwen_librispeech_pc_7_sampling_s0_t0.9_K2_V2_rw=0_pl=0_g=32_pre.wav")
    assert e["config"] == "K2V2@0+g32+pre" and (e["key_bits"], e["rw"], e["pl"]) == (2, 0, 0)
    assert parse_wav_name("qwen_x_7_sampling_s0_t0.9_K2_V2_rw=0_pl=0_g=32.wav")["config"] == "K2V2@0+g32"
    assert parse_wav_name("qwen_x_7_sampling_s0_t0.9_K2_V2_rw=0_pl=0_pre.wav")["config"] == "K2V2@0+pre"
    assert parse_wav_name("qwen_x_7_sampling_s0_t0.9_K2_V2_rw=0_pl=0.wav")["config"] == "K2V2@0"
    v = parse_wav_name("vallex_librispeech_pc_3_sampling_s0_K2V2@0+g32.wav")
    assert v["config"] == "K2V2@0+g32" and v["key_bits"] == 2
    assert parse_wav_name("vallex_x_3_sampling_s0_K2V2@0+g32-nar.wav")["config"] == "K2V2@0+g32-nar"


def test_check_audio_parses_keyfix_segments():
    from tools.check_audio import parse_wav_name

    d = parse_wav_name("qwen_librispeech_pc_7_sampling_s0_t0.9_K2_V2_rw=0_pl=0_g=32_pre.wav")
    assert d["config"] == "K2V2@0 pl0+g32+pre" and d["kb"] == 2
    assert parse_wav_name("qwen_x_7_sampling_s0_t0.9_K2_V2_rw=0_pl=0.wav")["config"] == "K2V2@0 pl0"
    assert parse_wav_name("vallex_x_3_sampling_s0_K2V2@0+g32.wav")["config"] == "K2V2@0+g32"
