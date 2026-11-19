"""Prompt protection (config.protect_prefix): cache, recorder, naming, scoring."""
from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

from tests.test_keyfix import (  # noqa: E402
    _cfg,
    _cols,
    _feed_qwen,
    _qwen_cache_mod,
    _recorder_ns,
    _single,
    _talker_module,
)
from turboquant.config import TurboQuantConfig  # noqa: E402


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------


def test_cache_prefix_auto_keeps_the_prefill_exact():
    mod = _qwen_cache_mod()
    cache = mod.TurboQuantKVCache(_cfg(protect_prefix=-1), n_layers=1)
    g = torch.Generator().manual_seed(0)
    P, D, H = 40, 64, 2
    k0, v0 = torch.randn(1, H, P, D, generator=g), torch.randn(1, H, P, D, generator=g)
    cache.update(k0, v0, layer_idx=0)  # prefill
    assert cache._prefix_len[0] == P
    k_in, k_out = _feed_qwen(cache, steps=4, step_len=8, seed=1)  # +32 decode tokens
    assert torch.equal(k_out[:, :, :P], k0)  # prompt keys untouched
    assert not torch.allclose(k_out[:, :, P:], k_in, atol=1e-3)  # generated keys lossy (rw=0)
    assert cache._compressed_upto[0] == P + 32
    # values protected too
    assert torch.equal(cache._buf_v[0][:, :, :P], v0)


def test_cache_prefix_fixed_count_and_off():
    mod = _qwen_cache_mod()
    cache = mod.TurboQuantKVCache(_cfg(protect_prefix=16), n_layers=1)
    k_in, k_out = _feed_qwen(cache, steps=5, step_len=8)  # 40 tokens, first call = 8
    assert cache._prefix_len[0] == 16
    assert torch.equal(k_out[:, :, :16], k_in[:, :, :16])
    assert not torch.allclose(k_out[:, :, 16:], k_in[:, :, 16:], atol=1e-3)
    off = mod.TurboQuantKVCache(_cfg(protect_prefix=0), n_layers=1)
    k_in2, k_out2 = _feed_qwen(off, steps=5, step_len=8)
    assert off._prefix_len[0] == 0 and not torch.allclose(k_out2[:, :, :8], k_in2[:, :, :8], atol=1e-3)


def test_cache_prefix_with_residual_window_and_groups():
    mod = _qwen_cache_mod()
    cache = mod.TurboQuantKVCache(_cfg(protect_prefix=-1, residual_window=8, key_group=16), n_layers=1)
    k0 = torch.randn(1, 2, 20, 64)
    cache.update(k0, k0.clone(), layer_idx=0)
    k_in, k_out = _feed_qwen(cache, steps=6, step_len=8, seed=2)  # 20 + 48 = 68 tokens
    # compressible region = [20, 60); groups of 16 from 20 -> [20, 52) compressed, 52..68 exact
    assert cache._compressed_upto[0] == 52
    assert torch.equal(k_out[:, :, :20], k0)
    assert torch.equal(k_out[:, :, 52:], k_in[:, :, 32:])
    assert not torch.allclose(k_out[:, :, 20:52], k_in[:, :, :32], atol=1e-3)


def test_cache_prefix_memory_report_counts_prompt_as_fp16():
    mod = _qwen_cache_mod()
    D, H, P, n = 64, 2, 40, 32
    on = mod.TurboQuantKVCache(_cfg(protect_prefix=-1, track_only=True), n_layers=1)
    off = mod.TurboQuantKVCache(_cfg(protect_prefix=0, track_only=True), n_layers=1)
    for c in (on, off):
        c.update(torch.randn(1, H, P, D), torch.randn(1, H, P, D), layer_idx=0)
        c.update(torch.randn(1, H, n, D), torch.randn(1, H, n, D), layer_idx=0)
    r_on, r_off = on.memory_report(), off.memory_report()
    assert r_on["fp16_equivalent_bytes"] == r_off["fp16_equivalent_bytes"]
    assert r_on["compressed_bytes"] > r_off["compressed_bytes"]
    # exactly P fp16 tokens more: K and V, 2 bytes each, minus the 2-bit storage they replace
    extra = r_on["compressed_bytes"] - r_off["compressed_bytes"]
    from turboquant.compressors_v3 import compressed_bytes_per_token

    expect = H * P * (2 * D * 2 - compressed_bytes_per_token(D, 2) - compressed_bytes_per_token(D, 2))
    assert extra == expect
    assert r_on["compression_ratio"] < r_off["compression_ratio"]


def test_config_default_off():
    assert TurboQuantConfig().protect_prefix == 0


# ---------------------------------------------------------------------------
# recorder
# ---------------------------------------------------------------------------


def test_recorder_prefix_auto_splices_prompt_and_measures_rest():
    ns = _recorder_ns()
    D, groups, P = 64, 2, 24
    rec = ns["DivergenceRecorder"](specs=[(2, 2, 0)], n_layers=4, prot_layers=0,
                                   prot_bits=8, stride=1, protect_prefix=-1)
    plain = ns["DivergenceRecorder"](specs=[(2, 2, 0)], n_layers=4, prot_layers=0,
                                     prot_bits=8, stride=1)
    rec.group = plain.group = "g"
    rec.idx = plain.idx = 0
    g = torch.Generator().manual_seed(0)
    key = torch.randn(1, 2, P, D, generator=g)
    q = torch.randn(1, 4, P, D, generator=g)
    rec.record(_talker_module(groups, P, 0, D), q, key, key, None, D**-0.5)  # prefill
    plain.record(_talker_module(groups, P, 0, D), q, key, key, None, D**-0.5)
    assert rec._prefix["SimpleNamespace"] == P
    d, dp = dict(zip(_cols(), rec.rows[0])), dict(zip(_cols(), plain.rows[0]))
    assert d["protect_prefix"] == -1 and dp["protect_prefix"] == 0
    assert d["attn_js"] < 1e-6 and d["cos_k"] == 1.0 and d["relmse_k"] == 0.0  # prompt only: exact
    assert dp["attn_js"] > 1e-3
    # decode steps: the prompt stays exact, the rest is measured
    key2 = torch.cat([key, torch.randn(1, 2, 16, D, generator=g)], dim=2)
    q2 = torch.randn(1, 4, 1, D, generator=g)
    rec.record(_talker_module(groups, 1, P, D), q2, key2, key2, None, D**-0.5)
    plain.record(_talker_module(groups, 1, P, D), q2, key2, key2, None, D**-0.5)
    d, dp = dict(zip(_cols(), rec.rows[1])), dict(zip(_cols(), plain.rows[1]))
    assert rec.errors == 0 and d["attn_js"] < dp["attn_js"]
    assert d["cos_k"] < 1.0  # the 16 generated keys are lossy and measured
    # recon on the compressed part only: same compressor, same tokens as plain's tail
    ns_keys = key2[:, :, P:]
    comp = next(iter(rec._cache.values()))
    rk, _ = comp.decompress_kv(*comp.compress_kv(key2, key2))
    cos_tail, _ = ns["_kv_recon_errors"](ns_keys, rk[:, :, P:], D)
    assert abs(d["cos_k"] - cos_tail) < 1e-6


def test_recorder_prefix_fixed_count():
    ns = _recorder_ns()
    D, groups = 64, 2
    rec = ns["DivergenceRecorder"](specs=[(2, 2, 0)], n_layers=4, prot_layers=0,
                                   prot_bits=8, stride=1, protect_prefix=8)
    rec.group, rec.idx = "g", 0
    key = torch.randn(1, 2, 32, D)
    rec.record(_talker_module(groups, 32, 0, D), torch.randn(1, 4, 32, D), key, key, None, D**-0.5)
    d = dict(zip(_cols(), rec.rows[0]))
    assert d["protect_prefix"] == 8 and 0.5 < d["cos_k"] < 1.0


# ---------------------------------------------------------------------------
# naming / resume / scoring
# ---------------------------------------------------------------------------


def test_make_config_and_wav_path_carry_protect_prefix(tmp_path):
    make_config = _single("make_config", {"TurboQuantConfig": TurboQuantConfig})
    assert make_config(2, 2, 0, 0, 0, False, -1).protect_prefix == -1
    assert make_config(2, 2, 0, 0).protect_prefix == 0
    wav_path = _single("_wav_path", {"os": os})
    assert wav_path(str(tmp_path), "g", 3, 0, 0.9, 2, 2, 0, 0, 0, False, -1).endswith("_pl=0_pp=prompt.wav")
    assert wav_path(str(tmp_path), "g", 3, 0, 0.9, 2, 2, 0, 0, 32, False, -1).endswith("_pl=0_g=32_pp=prompt.wav")
    assert wav_path(str(tmp_path), "g", 3, 0, 0.9, 2, 2, 0, 0, 0, False, 16).endswith("_pl=0_pp=16.wav")
    assert wav_path(str(tmp_path), "g", 3, 0, 0.9, 2, 2, 0, 0, 0, False, 0).endswith("_pl=0.wav")
    assert wav_path(str(tmp_path), "g", 3, 0, 0.9, None, None, None, 0, 0, False, -1).endswith("_fp16.wav")


def test_load_done_filters_protect_prefix(tmp_path):
    pd = pytest.importorskip("pandas")
    load_done = _single("load_done", {"pd": pd, "os": os})
    cols = _cols()
    rows = [
        ("g", 0, 0, 10, 2, 2, 0, 0, 0, 0, 0, "", 0) + tuple([0.1] * 9),
        ("g", 1, 0, 10, 2, 2, 0, 0, 0, 0, -1, "", 0) + tuple([0.1] * 9),
    ]
    p = tmp_path / "d.csv"
    pd.DataFrame(rows, columns=cols).to_csv(p, index=False)
    assert load_done(str(p), 0) == {("g", 0)}
    assert load_done(str(p), 0, 0, False, -1) == {("g", 1)}
    # a CSV without the column holds no protected rows
    old = tmp_path / "old.csv"
    pd.DataFrame(rows, columns=cols).drop(columns=["protect_prefix"]).to_csv(old, index=False)
    assert load_done(str(old), 0, 0, False, -1) == set()
    assert load_done(str(old), 0) == {("g", 0), ("g", 1)}


def test_score_and_check_audio_parse_pp_segment():
    import importlib.util

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def load(rel, name):
        spec = importlib.util.spec_from_file_location(name, os.path.join(root, rel))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    sw = load("tools/score_wav_dir.py", "score_wav_dir_pp")
    d = sw.parse_wav_name("qwen_librispeech_pc_2_sampling_s0_t0.9_K2_V2_rw=0_pl=0_g=32_pp=prompt.wav")
    assert d["config"] == "K2V2@0+g32+pp" and d["idx"] == 2
    d = sw.parse_wav_name("qwen_librispeech_pc_2_sampling_s0_t0.9_K3_V3_rw=0_pl=0_pp=16.wav")
    assert d["config"] == "K3V3@0+pp16"
    assert sw.parse_wav_name("qwen_librispeech_pc_2_sampling_s0_t0.9_K3_V3_rw=0_pl=0.wav")["config"] == "K3V3@0"
    try:
        ca = load("tools/check_audio.py", "check_audio_pp")
    except ModuleNotFoundError as exc:  # soundfile missing on a laptop venv
        pytest.skip(str(exc))
    m = ca._NAME_RE.match("qwen_librispeech_pc_2_sampling_s0_t0.9_K2_V2_rw=0_pl=0_pp=prompt.wav")
    assert m and m.group("pp") == "prompt"
