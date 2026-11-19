"""Prefix codebook (docs/superpowers plan 2026-11-18): fitted-levels stage."""

import types

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


from tests.test_keyfix import _cfg, _feed_qwen, _qwen_cache_mod  # noqa: E402


def test_cache_prefix_codebook_quantizes_the_prompt_and_leaves_generation_as_pp():
    mod = _qwen_cache_mod()
    g = torch.Generator().manual_seed(0)
    P, D, H = 40, 64, 2
    k0, v0 = torch.randn(1, H, P, D, generator=g), torch.randn(1, H, P, D, generator=g)
    pq = mod.TurboQuantKVCache(
        _cfg(protect_prefix=-1, prefix_codebook="kmeans", prefix_bits=3, key_group=16),
        n_layers=1,
    )
    pp = mod.TurboQuantKVCache(_cfg(protect_prefix=-1, key_group=16), n_layers=1)
    for c in (pq, pp):
        c.update(k0.clone(), v0.clone(), layer_idx=0)
    assert not torch.equal(pq._buf_k[0][:, :, :P], k0)  # prompt keys lossy
    assert not torch.equal(pq._buf_v[0][:, :, :P], v0)  # prompt values quantized too
    cos = torch.nn.functional.cosine_similarity(
        pq._buf_k[0][:, :, :P].flatten(2), k0.flatten(2), dim=-1
    ).mean()
    assert cos > 0.97
    assert pq._chunks_k[0][0].get("prefix") is True and pq._chunks_k[0][0]["span"] == P
    k_in_q, k_out_q = _feed_qwen(pq, steps=4, step_len=8, seed=1)
    k_in_p, k_out_p = _feed_qwen(pp, steps=4, step_len=8, seed=1)
    assert torch.equal(
        k_out_q[:, :, P:], k_out_p[:, :, P:]
    )  # generated path identical to pp
    assert pq._compressed_upto[0] == pp._compressed_upto[0] == P + 32


def test_cache_prefix_codebook_with_identity_rope_matches_post_rope_path():
    # cos = 1, sin = 0 makes RoPE the identity, so the pre-RoPE path must reproduce the plain path
    mod = _qwen_cache_mod()
    P, D, H = 40, 64, 2
    g = torch.Generator().manual_seed(0)
    k0, v0 = torch.randn(1, H, P, D, generator=g), torch.randn(1, H, P, D, generator=g)
    plain = mod.TurboQuantKVCache(
        _cfg(protect_prefix=-1, prefix_codebook="kmeans", prefix_bits=3), n_layers=1
    )
    pre = mod.TurboQuantKVCache(
        _cfg(
            protect_prefix=-1,
            prefix_codebook="kmeans",
            prefix_bits=3,
            pre_rope_keys=True,
        ),
        n_layers=1,
        rope_scaling=None,
    )
    plain.update(k0.clone(), v0.clone(), layer_idx=0)
    pre.update(
        k0.clone(),
        v0.clone(),
        layer_idx=0,
        cache_kwargs={"cos": torch.ones(1, P, D), "sin": torch.zeros(1, P, D)},
    )
    assert torch.allclose(pre._buf_k[0][:, :, :P], plain._buf_k[0][:, :, :P], atol=1e-5)
    assert not torch.equal(pre._buf_k[0][:, :, :P], k0)


def test_cache_prefix_codebook_memory_report():
    mod = _qwen_cache_mod()
    D, H, P, n = 64, 2, 40, 32
    pq = mod.TurboQuantKVCache(
        _cfg(
            protect_prefix=-1, prefix_codebook="minmax", prefix_bits=3, track_only=True
        ),
        n_layers=1,
    )
    pp = mod.TurboQuantKVCache(_cfg(protect_prefix=-1, track_only=True), n_layers=1)
    for c in (pq, pp):
        c.update(torch.randn(1, H, P, D), torch.randn(1, H, P, D), layer_idx=0)
        c.update(torch.randn(1, H, n, D), torch.randn(1, H, n, D), layer_idx=0)
    diff = (
        pp.memory_report()["compressed_bytes"] - pq.memory_report()["compressed_bytes"]
    )
    expect = (
        H
        * P
        * (
            2 * D * 2
            - compressed_bytes_per_token(D, 3, 0, 2, P)
            - compressed_bytes_per_token(D, 2)
        )
    )
    assert abs(diff - expect) < 1
    assert torch.isfinite(pq._buf_k[0][:, :, :P]).all()  # track_only: buffer untouched


def test_cache_default_config_unchanged():
    mod = _qwen_cache_mod()
    a, b = (
        mod.TurboQuantKVCache(_cfg(), n_layers=1),
        mod.TurboQuantKVCache(_cfg(prefix_codebook=""), n_layers=1),
    )
    ka, oa = _feed_qwen(a, steps=5, step_len=8)
    kb, ob = _feed_qwen(b, steps=5, step_len=8)
    assert torch.equal(oa, ob)


def test_recorder_prefix_codebook_splices_quantized_prompt():
    import os  # noqa: F401
    from tests.test_keyfix import _cols, _recorder_ns, _talker_module

    ns = _recorder_ns()
    D, groups, P = 64, 2, 24
    rec = ns["DivergenceRecorder"](
        specs=[(2, 2, 0)],
        n_layers=4,
        prot_layers=0,
        prot_bits=8,
        stride=1,
        protect_prefix=-1,
        prefix_codebook="kmeans",
        prefix_bits=3,
    )
    pp = ns["DivergenceRecorder"](
        specs=[(2, 2, 0)],
        n_layers=4,
        prot_layers=0,
        prot_bits=8,
        stride=1,
        protect_prefix=-1,
    )
    for r in (rec, pp):
        r.group, r.idx = "g", 0
    g = torch.Generator().manual_seed(0)
    key = torch.randn(1, 2, P, D, generator=g)
    q = torch.randn(1, 4, P, D, generator=g)
    for r in (rec, pp):
        r.record(_talker_module(groups, P, 0, D), q, key, key, None, D**-0.5)
    d, dp = dict(zip(_cols(), rec.rows[0])), dict(zip(_cols(), pp.rows[0]))
    assert (
        d["prefix_codebook"] == "kmeans"
        and d["prefix_bits"] == 3
        and dp["prefix_codebook"] == ""
    )
    assert (
        0 < d["attn_js"] and d["cos_k"] < 1.0 and d["relmse_k"] > 0
    )  # prompt lossy, measured
    assert dp["attn_js"] < 1e-6  # fp16 prompt still exact
    assert rec.errors == 0


def test_recorder_compressor_cache_key_includes_effective_prefix_bits():
    from tests.test_keyfix import _recorder_ns

    ns = _recorder_ns()
    rec = ns["DivergenceRecorder"](
        specs=[(8, 8, 0), (2, 2, 0)],
        n_layers=4,
        prot_layers=0,
        prot_bits=8,
        stride=1,
        protect_prefix=-1,
        prefix_codebook="kmeans",
        prefix_bits=0,
    )
    a = rec._comp(0, 8, 8, 0, 16, "cpu")
    b = rec._comp(0, 2, 2, 0, 16, "cpu")
    assert a.prefix_bits_effective == 8 and b.prefix_bits_effective == 2


def test_wav_path_and_make_config_carry_codebook(tmp_path):
    import os

    from tests.test_keyfix import _single

    wav_path = _single("_wav_path", {"os": os})
    assert wav_path(
        str(tmp_path), "g", 3, 0, 0.9, 2, 2, 0, 0, 32, True, -1, "kmeans", 3
    ).endswith("_pl=0_g=32_pre_pq=k3.wav")
    assert wav_path(
        str(tmp_path), "g", 3, 0, 0.9, 8, 8, 0, 0, 32, True, -1, "minmax", 0
    ).endswith("_pq=m8.wav")
    assert wav_path(str(tmp_path), "g", 3, 0, 0.9, 2, 2, 0, 0, 32, True, -1).endswith(
        "_pp=prompt.wav"
    )
    assert wav_path(
        str(tmp_path), "g", 3, 0, 0.9, None, None, None, 0, 0, False, -1, "kmeans", 2
    ).endswith("_fp16.wav")
    make_config = _single("make_config", {"TurboQuantConfig": TurboQuantConfig})
    c = make_config(2, 2, 0, 0, 32, True, -1, "kmeans", 2)
    assert c.prefix_codebook == "kmeans" and c.prefix_bits == 2


def test_load_done_filters_codebook_and_old_csv(tmp_path):
    import os

    from tests.test_keyfix import _cols, _single

    pd = pytest.importorskip("pandas")
    load_done = _single("load_done", {"pd": pd, "os": os})
    cols = _cols()
    rows = [
        ("g", 0, 0, 10, 2, 2, 0, 0, 32, 1, -1, "", 0) + tuple([0.1] * 9),
        ("g", 1, 0, 10, 2, 2, 0, 0, 32, 1, -1, "kmeans", 2) + tuple([0.1] * 9),
    ]
    p = tmp_path / "d.csv"
    pd.DataFrame(rows, columns=cols).to_csv(p, index=False)
    assert load_done(str(p), 0, 32, True, -1) == {("g", 0)}
    assert load_done(str(p), 0, 32, True, -1, "kmeans", 2) == {("g", 1)}
    old = tmp_path / "old.csv"
    pd.DataFrame(rows, columns=cols).drop(
        columns=["prefix_codebook", "prefix_bits"]
    ).to_csv(old, index=False)
    assert load_done(str(old), 0, 32, True, -1, "kmeans", 2) == set()


def test_recorder_prefix_codebook_pre_rope_decode_step_records():
    from tests.test_keyfix import _cols, _recorder_ns, _talker_module

    ns = _recorder_ns()
    D, groups, P = 64, 2, 24
    rec = ns["DivergenceRecorder"](
        specs=[(2, 2, 0)],
        n_layers=4,
        prot_layers=0,
        prot_bits=8,
        stride=1,
        pre_rope=True,
        protect_prefix=-1,
        prefix_codebook="kmeans",
        prefix_bits=3,
    )
    rec.group, rec.idx = "g", 0
    g = torch.Generator().manual_seed(0)
    key = torch.randn(1, 2, P + 1, D, generator=g)
    q = torch.randn(1, 4, P, D, generator=g)
    rec.record(
        _talker_module(groups, P, 0, D), q, key[:, :, :P], key[:, :, :P], None, D**-0.5
    )
    q1 = torch.randn(1, 4, 1, D, generator=g)
    rec.record(_talker_module(groups, 1, P, D), q1, key, key, None, D**-0.5)
    assert rec.errors == 0, rec.first_error
    assert [dict(zip(_cols(), r))["pos"] for r in rec.rows] == [P, P + 1]


def test_summarize_label_with_codebook():
    from tools import summarize_keyfix as sm

    assert sm._label(2, 2, 0, 32, 1, -1, "kmeans", 2) == "K2V2@0+g32+pre+pk2"
    assert sm._label(2, 2, 0, 32, 1, -1) == "K2V2@0+g32+pre+pp"


def test_effective_bits_with_prefix_cost():
    import numpy as np

    from tools import analyze_keyfix as an

    P, S = np.array([53.0]), np.array([303.0])
    got = an.effective_bits(2, 1.0, True, P, S, prefix_bits_per_elem=3.2)
    assert got == pytest.approx((3.2 * 53 + 3 * 250) / 303)
    assert an.effective_bits(2, 1.0, True, P, S) == pytest.approx(
        (16 * 53 + 3 * 250) / 303
    )


def test_runaway_classification_counts():
    import pandas as pd

    from tools import analyze_keyfix as an

    scores = pd.DataFrame(
        {
            "idx": [0, 0, 1, 1],
            "config": ["fp16", "K2V2@0+g32", "fp16", "K2V2@0+g32"],
            "dur_s": [4.0, 30.0, 5.0, 5.2],
            "cer": [0.0, 0.9, 0.0, 0.6],
            "spk_sim": [1, 0.8, 1, 0.9],
        }
    )
    r = an.runaway_table({"g32": scores})
    row = r[(r.cell == "g32") & (r.config == "K2V2@0")].iloc[0]
    assert row.n_collapse == 2 and row.runaway == 1 and row.stopped == 1


def test_step_table_auc_separates_collapsed_from_ok():
    import pandas as pd

    from tools import analyze_keyfix as an

    scores = pd.DataFrame(
        {
            "idx": [0, 1, 2, 3],
            "config": ["K3V3@0"] * 4,
            "cer": [0.9, 0.8, 0.0, 0.1],
            "dur_s": [9.0] * 4,
            "spk_sim": [1.0] * 4,
        }
    )
    step = pd.DataFrame(
        {
            "idx": [0, 1, 2, 3],
            "layer": [5] * 4,
            "key_bits": [3] * 4,
            "value_bits": [3] * 4,
            "rw": [0] * 4,
            "js_max": [0.9, 0.8, 0.1, 0.2],
            "js_p95": [0.5, 0.4, 0.1, 0.2],
        }
    )
    t = an.step_table({"g32": step}, {"g32": scores})
    assert len(t) == 1 and t.iloc[0].auc_js_max == 1.0


def test_validator_wav_name_for_pk():
    import importlib.util
    import os

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "validate_pk", os.path.join(root, "tools/validate_keyfix_smoke.py")
    )
    v = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v)
    assert v._wav("w", 1, (2, 2, "pk")).endswith("_K2_V2_rw=0_pl=0_g=32_pre_pq=k2.wav")
    assert v._wav("w", 1, (8, 8, "pk")).endswith("_pq=k8.wav")
    assert v._wav("w", 1, (2, 2, "pp")).endswith("_pp=prompt.wav")
    assert v.CELLS["pk"] == (32, 1, -1, "kmeans", 0)
    assert v.CELLS["plain"] == (0, 0, 0, "", 0)


def test_prefix_cell_bits_counts_codebook_keys_and_quantized_values():
    import pandas as pd

    from tools import analyze_keyfix as an

    frame = pd.DataFrame(
        {
            "group": ["g"],
            "idx": [0],
            "key_bits": [2],
            "value_bits": [2],
            "rw": [0],
            "protected_layers": [0],
            "key_group": [32],
            "pre_rope": [1],
            "protect_prefix": [-1],
            "prefix_codebook": ["kmeans"],
            "prefix_bits": [2],
            "prompt_len": [53],
            "seq_len": [303],
            "n_steps": [250],
        }
    )
    got = an.prefix_cell_bits(frame, 2, 2)
    assert got["key"] == pytest.approx(((2 + 4 * 16 / 53) * 53 + 3 * 250) / 303)
    assert got["value"] == pytest.approx(2.0)


def test_check_resume_header(tmp_path):
    import pandas as pd

    from tests.test_keyfix import _single

    check = _single("check_resume_header", {"pd": pd, "COLUMNS": ["a", "b"]})
    old, ok = tmp_path / "old.csv", tmp_path / "ok.csv"
    old.write_text("a\n1\n")
    ok.write_text("a,b\n1,2\n")
    check(str(ok))
    with pytest.raises(SystemExit):
        check(str(old))


def test_recorder_memoizes_prompt_round_trip():
    from tests.test_keyfix import _recorder_ns, _talker_module

    ns = _recorder_ns()
    D, P = 64, 24
    rec = ns["DivergenceRecorder"](
        specs=[(2, 2, 0)], n_layers=4, prot_layers=0, prot_bits=8, stride=1,
        protect_prefix=-1, prefix_codebook="kmeans", prefix_bits=3,
    )  # fmt: skip
    rec.group, rec.idx = "g", 0
    g = torch.Generator().manual_seed(0)
    key = torch.randn(1, 2, P + 2, D, generator=g)
    q = torch.randn(1, 4, P, D, generator=g)
    rec.record(_talker_module(2, P, 0, D), q, key[:, :, :P], key[:, :, :P], None, D**-0.5)
    (entry,) = rec._prompt_rt.values()
    rec.record(
        _talker_module(2, 1, P, D), q[:, :, :1], key[:, :, : P + 1], key[:, :, : P + 1],
        None, D**-0.5,
    )  # fmt: skip
    assert len(rec._prompt_rt) == 1
    assert next(iter(rec._prompt_rt.values()))[2] is entry[2]  # same round trip reused
    assert rec.errors == 0


@pytest.mark.parametrize("mode", ["kmeans", "minmax"])
@pytest.mark.parametrize("pre_rope", [True, False])
def test_cache_and_recorder_quantize_prompt_identically(mode, pre_rope):
    from tests.test_keyfix import (
        _MROPE,
        _cfg,
        _qwen_cache_mod,
        _recorder_ns,
        _rope_kwargs_for_positions,
    )

    mod, ns = _qwen_cache_mod(), _recorder_ns()
    D, H, P = 64, 2, 40
    g = torch.Generator().manual_seed(0)
    k0 = torch.randn(1, H, P, D, generator=g)
    v0 = torch.randn(1, H, P, D, generator=g)
    cfg = _cfg(
        protect_prefix=-1, prefix_codebook=mode, prefix_bits=3, key_group=16,
        pre_rope_keys=pre_rope,
    )  # fmt: skip
    cache = mod.TurboQuantKVCache(cfg, n_layers=4, rope_scaling=_MROPE)
    kw = _rope_kwargs_for_positions(0, P, D, _MROPE)
    kwargs = {"cos": kw["cos"], "sin": kw["sin"]} if pre_rope else None
    cache.update(k0.clone(), v0.clone(), layer_idx=1, cache_kwargs=kwargs)
    ck, cv = cache._buf_k[1][:, :, :P].clone(), cache._buf_v[1][:, :, :P].clone()
    Rec = ns["DivergenceRecorder"]
    rec = Rec(
        specs=[(2, 2, 0)], n_layers=4, prot_layers=0, prot_bits=8, stride=1, key_group=16,
        pre_rope=pre_rope, protect_prefix=-1, prefix_codebook=mode, prefix_bits=3,
    )  # fmt: skip
    comp = rec._comp(1, 2, 2, 0, D, "cpu")
    rope, key_in = None, k0
    if pre_rope:
        cos, sin = kw["cos"], kw["sin"]
        rope = dict(
            cos=cos, sin=sin,
            mrope_section=_MROPE.get("mrope_section") if cos.dim() == 4 else None,
            interleaved=bool(_MROPE.get("interleaved", False)),
        )  # fmt: skip
        key_in = ns["rope_keys"](k0, inverse=True, **rope)
    module = types.SimpleNamespace(layer_idx=1)
    rk, rv = rec._splice_quantized_prefix(
        module, comp, key_in, v0, k0.clone(), v0.clone(), P, rope, k0.dtype, (2, 2, 0)
    )
    assert not torch.equal(ck, k0)  # the cache really quantized the prompt
    assert torch.equal(rk, ck) and torch.equal(rv, cv)
