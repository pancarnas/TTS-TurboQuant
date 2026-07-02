"""Tests for kv_attn_divergence_experiment.

Pure helpers and the measurement core are exec-extracted from the source with
their dependencies injected, so the heavy model imports (qwen_tts, Qwen3TTSModel)
are never triggered and everything runs on CPU.
"""

from __future__ import annotations

import os
import re
import traceback
import types

import pandas as pd
import pytest
import torch

from turboquant.compressors_v3 import TurboQuantV3
from turboquant.config import TurboQuantConfig

_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "Qwen3-TTS",
    "benchmarks",
    "kv_attn_divergence_experiment.py",
)
_SRC = open(_PATH, encoding="utf-8").read()

# Column order the recorder appends in (mirrors COLUMNS in the module).
COLS = [
    "group",
    "idx",
    "layer",
    "pos",
    "key_bits",
    "value_bits",
    "rw",
    "protected_layers",
    "attn_js",
    "attn_tv",
    "attn_top1",
    "attn_dentropy",
    "out_cos",
    "cos_k",
    "cos_v",
    "relmse_k",
    "relmse_v",
]
_METRIC_NAMES = COLS[8:]


def _repeat_kv(hidden_states, n_rep):
    b, nkv, slen, hd = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hs = hidden_states[:, :, None, :, :].expand(b, nkv, n_rep, slen, hd)
    return hs.reshape(b, nkv * n_rep, slen, hd)


def _kv_recon_errors(orig, recon, head_dim):
    a = orig.reshape(-1, head_dim).float()
    b = recon.reshape(-1, head_dim).float()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item()
    relmse = (((a - b) ** 2).sum() / (a**2).sum().clamp_min(1e-12)).item()
    return cos, relmse


def _config_label(kb, vb, rw):
    return f"K{kb}V{vb}@{rw}"


def _block():
    """Exec js_divergence..make_patch with deps injected; return the namespace."""
    start = _SRC.index("def js_divergence(")
    end = _SRC.index("\ndef force_eager(")
    ns = {
        "torch": torch,
        "traceback": traceback,
        "modeling": types.SimpleNamespace(repeat_kv=_repeat_kv),
        "TurboQuantV3": TurboQuantV3,
        "_kv_recon_errors": _kv_recon_errors,
    }
    exec(compile(_SRC[start:end], _PATH, "exec"), ns)
    return ns


def _single_func(name, ns_extra):
    start = _SRC.index(f"def {name}(")
    end = _SRC.index("\ndef ", start + 1)
    ns = dict(ns_extra)
    exec(compile(_SRC[start:end], _PATH, "exec"), ns)
    return ns[name]


# --- config parsing -------------------------------------------------------


def test_parse_configs():
    parse = _single_func(
        "parse_configs",
        {"re": re, "_CFG_RE": re.compile(r"[Kk](\d+)[Vv](\d+)@(\d+)$")},
    )
    assert parse("K4V4@24,K4V3@24,K3V3@24,K4V4@0") == [
        (4, 4, 24),
        (4, 3, 24),
        (3, 3, 24),
        (4, 4, 0),
    ]


# --- pure metric helpers --------------------------------------------------


def test_js_divergence_identical_disjoint_symmetric():
    js = _single_func("js_divergence", {"torch": torch})
    p = torch.tensor([[0.2, 0.3, 0.5]])
    assert js(p, p.clone()) < 1e-6
    assert abs(js(torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]])) - 1.0) < 1e-3
    a, b = torch.tensor([[0.7, 0.2, 0.1]]), torch.tensor([[0.1, 0.3, 0.6]])
    assert abs(js(a, b) - js(b, a)) < 1e-6 and 0.0 <= js(a, b) <= 1.0


def test_total_variation_bounds():
    tv = _single_func("total_variation", {"torch": torch})
    p = torch.tensor([[0.2, 0.3, 0.5]])
    assert tv(p, p.clone()) < 1e-6
    assert abs(tv(torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]])) - 1.0) < 1e-6


def test_top1_agreement():
    top1 = _single_func("top1_agreement", {"torch": torch})
    p = torch.tensor([[0.1, 0.7, 0.2], [0.6, 0.3, 0.1]])
    same = torch.tensor([[0.2, 0.6, 0.2], [0.5, 0.4, 0.1]])
    flip = torch.tensor([[0.7, 0.1, 0.2], [0.1, 0.1, 0.8]])
    assert top1(p, same) == 1.0 and top1(p, flip) == 0.0


def test_entropy_delta_sign():
    de = _single_func("entropy_delta", {"torch": torch})
    peaked = torch.tensor([[0.97, 0.01, 0.01, 0.01]])
    uniform = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
    assert de(peaked, uniform) > 0 and de(uniform, peaked) < 0


def test_output_cosine():
    oc = _single_func("output_cosine", {"torch": torch})
    v = torch.randn(1, 2, 1, 8)
    assert abs(oc(v, v.clone()) - 1.0) < 1e-6 and abs(oc(v, -v) + 1.0) < 1e-6


# --- streaming aggregate + summarize (config-keyed) -----------------------


def test_streaming_aggregate_matches_means_and_diff():
    acc = _single_func("accumulate_running", {"_METRICS": _METRIC_NAMES})
    frame = _single_func(
        "running_frame",
        {"_METRICS": _METRIC_NAMES, "pd": pd, "config_label": _config_label},
    )

    def _row(kb, vb, rw, v):  # (group, idx, layer, pos, kb, vb, rw, pl, *9 metrics)
        return ("g", 0, 0, 10, kb, vb, rw, 2) + tuple([v] * 9)

    running: dict = {}
    acc(running, [_row(4, 4, 24, 0.01), _row(4, 4, 0, 0.20)])
    acc(running, [_row(4, 4, 24, 0.03), _row(4, 4, 0, 0.30)])
    assert running[(4, 4, 24)]["_n"] == 2 and running[(4, 4, 0)]["_n"] == 2
    out = frame(running)
    assert abs(out.loc["K4V4@24", "attn_js"] - 0.02) < 1e-9
    assert abs(out.loc["K4V4@0", "attn_js"] - 0.25) < 1e-9
    # Δ(0-24) row present for K4V4 (both windows seen)
    assert abs(out.loc["K4V4 d(0-24)", "attn_js"] - 0.23) < 1e-9


def test_summarize_groups_by_config():
    summarize = _single_func("summarize", {"_METRICS": _METRIC_NAMES})
    rows = [
        ("g", 0, 0, 10, 4, 4, 24, 2) + tuple([0.02] * 9),
        ("g", 1, 0, 10, 4, 2, 24, 2) + tuple([0.20] * 9),
        ("g", 1, 0, 10, 4, 2, 24, 0) + tuple([0.50] * 9),  # other pl -> own bucket
    ]
    df = pd.DataFrame(rows, columns=COLS)
    out = summarize(df)
    assert abs(out.loc[(4, 4, 24, 2), "attn_js"] - 0.02) < 1e-9
    assert abs(out.loc[(4, 2, 24, 2), "attn_js"] - 0.20) < 1e-9
    assert abs(out.loc[(4, 2, 24, 0), "attn_js"] - 0.50) < 1e-9


# --- force_eager (wrapper traversal) --------------------------------------


def test_force_eager_walks_wrapped_model():
    force_eager = _single_func("force_eager", {"torch": torch})

    class Attn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = types.SimpleNamespace(_attn_implementation="sdpa")

    class Talker(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = Attn()

    class Wrapper:  # NOT an nn.Module, like Qwen3TTSModel
        def __init__(self):
            self.model = Talker()

    w = Wrapper()
    force_eager(w)
    assert w.model.attn.config._attn_implementation == "eager"


# --- measurement core (CPU, real TurboQuantV3) ----------------------------


def _attn_inputs(seq, head_dim=64, nkv=2, groups=2):
    torch.manual_seed(0)
    q = torch.randn(1, nkv * groups, 1, head_dim)
    key = torch.randn(1, nkv, seq, head_dim)
    value = torch.randn(1, nkv, seq, head_dim)
    return q, key, value, head_dim, groups


def _module(groups, layer_idx=0):
    return types.SimpleNamespace(layer_idx=layer_idx, num_key_value_groups=groups)


def test_compressed_attention_is_a_distribution():
    ns = _block()
    q, key, _, hd, groups = _attn_inputs(32)
    aw = ns["compressed_attention"](q, key, groups, None, hd**-0.5)
    assert aw.shape == (1, groups * 2, 1, 32)
    assert torch.allclose(aw.sum(-1), torch.ones_like(aw.sum(-1)), atol=1e-4)


def test_recorder_logs_one_row_per_config_with_sane_values():
    ns = _block()
    q, key, value, hd, groups = _attn_inputs(64)
    scaling = hd**-0.5
    rec = ns["DivergenceRecorder"](
        specs=[(4, 4, 16), (4, 4, 0)],
        n_layers=4,
        prot_layers=0,
        prot_bits=8,
        stride=1,
    )
    rec.group, rec.idx = "g", 0
    rec.record(_module(groups), q, key, value, None, scaling)

    assert rec.errors == 0 and len(rec.rows) == 2  # one per config
    by_rw = {r[6]: dict(zip(COLS, r)) for r in rec.rows}  # r[6] = rw
    for d in by_rw.values():
        assert d["pos"] == 64 and 0.0 <= d["attn_js"] <= 1.0 + 1e-6
        assert d["cos_k"] <= 1.0 + 1e-6 and d["relmse_k"] >= 0.0
    # rw=0 quantizes ALL tokens; rw=16 keeps last 16 exact -> rw=0 worse
    assert by_rw[0]["relmse_k"] >= by_rw[16]["relmse_k"]
    assert by_rw[0]["attn_js"] >= by_rw[16]["attn_js"]
    assert by_rw[0]["cos_k"] <= by_rw[16]["cos_k"]
    assert by_rw[0]["out_cos"] <= by_rw[16]["out_cos"]


def test_recorder_step_stride_skips_unrecorded_positions():
    ns = _block()
    rec = ns["DivergenceRecorder"](
        specs=[(4, 4, 0)], n_layers=4, prot_layers=0, prot_bits=8, stride=4
    )
    rec.group, rec.idx = "g", 0
    scaling = 64**-0.5
    q, key, value, hd, groups = _attn_inputs(63)  # 63 % 4 != 0 -> skipped
    rec.record(_module(groups), q, key, value, None, scaling)
    assert rec.rows == []
    q, key, value, hd, groups = _attn_inputs(64)  # 64 % 4 == 0 -> recorded
    rec.record(_module(groups), q, key, value, None, scaling)
    assert len(rec.rows) == 1


def test_recorder_reuses_compressors_per_config():
    ns = _block()
    rec = ns["DivergenceRecorder"](
        specs=[(4, 4, 16), (4, 4, 0)],
        n_layers=4,
        prot_layers=0,
        prot_bits=8,
        stride=1,
    )
    rec.group, rec.idx = "g", 0
    scaling = 64**-0.5
    for _ in range(3):
        q, key, value, hd, groups = _attn_inputs(64)
        rec.record(_module(groups), q, key, value, None, scaling)
    assert len(rec._cache) == 2  # one TurboQuantV3 per (layer, kb, vb, rw)


def test_resolve_n_layers_reads_talker_config_and_fails_on_missing():
    fn = _single_func("resolve_n_layers", {})

    class Good:
        class model:
            class config:
                class talker_config:
                    num_hidden_layers = 20

    assert fn(Good) == 20  # reads talker_config, not the (absent) top-level attr

    class Bad:  # no resolvable layer count -> must refuse, not silently return 0
        class model:
            class config:
                pass

    with pytest.raises(SystemExit):
        fn(Bad)


def test_recorder_bits_matter_on_unprotected_layer():
    # THE regression for the n_layers=0 bug: with a real layer count, K4V2 must
    # distort values MORE than K4V4 on a non-protected layer. If every layer were
    # forced to 8-bit (the bug), these would be byte-identical.
    ns = _block()
    q, key, value, hd, groups = _attn_inputs(64)
    scaling = hd**-0.5
    rec = ns["DivergenceRecorder"](
        specs=[(4, 4, 24), (4, 2, 24)],
        n_layers=20,
        prot_layers=2,
        prot_bits=8,
        stride=1,
    )
    rec.group, rec.idx = "g", 0
    rec.record(_module(groups, layer_idx=10), q, key, value, None, scaling)
    by = {(r[4], r[5], r[6]): dict(zip(COLS, r)) for r in rec.rows}
    assert by[(4, 2, 24)]["relmse_v"] > by[(4, 4, 24)]["relmse_v"]


def test_recorder_protected_layer_forces_same_bits():
    # On a protected edge layer both configs use protected_bits -> identical.
    ns = _block()
    q, key, value, hd, groups = _attn_inputs(64)
    scaling = hd**-0.5
    rec = ns["DivergenceRecorder"](
        specs=[(4, 4, 24), (4, 2, 24)],
        n_layers=20,
        prot_layers=2,
        prot_bits=8,
        stride=1,
    )
    rec.group, rec.idx = "g", 0
    rec.record(
        _module(groups, layer_idx=0), q, key, value, None, scaling
    )  # layer 0 protected
    by = {(r[4], r[5], r[6]): dict(zip(COLS, r)) for r in rec.rows}
    assert by[(4, 2, 24)]["relmse_v"] == by[(4, 4, 24)]["relmse_v"]


def test_recorder_exact_zero_when_nothing_compressed():
    # rw >= S: compress_kv returns the fp16 K/V untouched, and because the
    # reference map is computed through the SAME fp32 path as the compressed
    # map, the row must be exact (no bf16-rounding noise floor).
    ns = _block()
    q, key, value, hd, groups = _attn_inputs(32)
    scaling = hd**-0.5
    rec = ns["DivergenceRecorder"](
        specs=[(4, 4, 128)], n_layers=4, prot_layers=0, prot_bits=8, stride=1
    )
    rec.group, rec.idx = "g", 0
    rec.record(_module(groups), q, key, value, None, scaling)
    d = dict(zip(COLS, rec.rows[0]))
    assert d["attn_js"] == 0.0 and d["attn_tv"] == 0.0
    assert d["attn_top1"] == 1.0 and d["attn_dentropy"] == 0.0
    assert d["out_cos"] >= 1.0 - 1e-6
    assert d["cos_k"] == 1.0 and d["cos_v"] == 1.0
    assert d["relmse_k"] == 0.0 and d["relmse_v"] == 0.0


def test_recorder_recon_metrics_use_compressed_prefix_only():
    # With 0 < rw < S the exact fp16 tail must be excluded from cos/relmse —
    # the row must equal _kv_recon_errors on the first S-rw tokens only.
    ns = _block()
    seq, rw = 64, 16
    q, key, value, hd, groups = _attn_inputs(seq)
    scaling = hd**-0.5
    rec = ns["DivergenceRecorder"](
        specs=[(4, 4, rw)], n_layers=4, prot_layers=0, prot_bits=8, stride=1
    )
    rec.group, rec.idx = "g", 0
    rec.record(_module(groups), q, key, value, None, scaling)
    d = dict(zip(COLS, rec.rows[0]))

    comp = rec._cache[(0, 4, 4, rw)]  # the exact compressor the recorder used
    rk, rv = comp.decompress_kv(*comp.compress_kv(key, value))
    split = seq - rw
    exp_cos_k, exp_relmse_k = _kv_recon_errors(
        key[:, :, :split], rk[:, :, :split], hd
    )
    exp_cos_v, exp_relmse_v = _kv_recon_errors(
        value[:, :, :split], rv[:, :, :split], hd
    )
    assert abs(d["cos_k"] - exp_cos_k) < 1e-9 and abs(d["cos_v"] - exp_cos_v) < 1e-9
    assert abs(d["relmse_k"] - exp_relmse_k) < 1e-9
    assert abs(d["relmse_v"] - exp_relmse_v) < 1e-9
    # sanity: the prefix genuinely has quantization error (dilution would hide it)
    assert d["relmse_k"] > 0.0 and d["cos_k"] < 1.0


def test_recorder_rows_carry_protected_layers():
    ns = _block()
    q, key, value, hd, groups = _attn_inputs(32)
    scaling = hd**-0.5
    rec = ns["DivergenceRecorder"](
        specs=[(4, 4, 0)], n_layers=20, prot_layers=3, prot_bits=8, stride=1
    )
    rec.group, rec.idx = "g", 0
    rec.record(_module(groups, layer_idx=10), q, key, value, None, scaling)
    d = dict(zip(COLS, rec.rows[0]))
    assert d["protected_layers"] == 3 and d["rw"] == 0


def test_make_config_threads_protected_layers():
    make_config = _single_func("make_config", {"TurboQuantConfig": TurboQuantConfig})
    c = make_config(4, 2, 24, 0)
    assert c.protected_layers == 0 and c.track_only is False
    assert (c.key_bits, c.value_bits, c.residual_window) == (4, 2, 24)
    assert make_config(4, 2, 24).protected_layers == 2  # default preserved
    # protected_layers=0 -> uniform bits: even layer 0 keeps the requested bits
    tq = TurboQuantV3(
        head_dim=8, key_bits=4, value_bits=2, layer_idx=0, n_layers=20,
        protected_layers=0, protected_bits=5,
    )
    assert (tq.key_bits, tq.value_bits) == (4, 2)
    tq_prot = TurboQuantV3(
        head_dim=8, key_bits=4, value_bits=2, layer_idx=0, n_layers=20,
        protected_layers=2, protected_bits=5,
    )
    assert (tq_prot.key_bits, tq_prot.value_bits) == (5, 5)


def test_load_done_reads_group_idx_pairs(tmp_path):
    load_done = _single_func("load_done", {"pd": pd, "os": os})
    p = tmp_path / "k.csv"
    assert load_done(str(p), 2) == set()  # missing -> empty
    rows = [
        ("g", 0, 0, 10, 4, 4, 24, 2) + tuple([0.1] * 9),
        ("g", 0, 1, 10, 4, 2, 24, 2) + tuple([0.2] * 9),  # same sentence, other config
        ("g", 1, 0, 10, 4, 4, 24, 2) + tuple([0.3] * 9),
        ("g", 2, 0, 10, 4, 4, 24, 0) + tuple([0.4] * 9),  # different protected_layers
    ]
    pd.DataFrame(rows, columns=COLS).to_csv(p, index=False)
    # only rows recorded under the SAME protected_layers count as done
    assert load_done(str(p), 2) == {("g", 0), ("g", 1)}
    assert load_done(str(p), 0) == {("g", 2)}


def test_load_done_refuses_prefix_csv_without_protected_layers(tmp_path):
    load_done = _single_func("load_done", {"pd": pd, "os": os})
    p = tmp_path / "old.csv"
    old_cols = [c for c in COLS if c != "protected_layers"]
    rows = [("g", 0, 0, 10, 4, 4, 24) + tuple([0.1] * 9)]
    pd.DataFrame(rows, columns=old_cols).to_csv(p, index=False)
    with pytest.raises(SystemExit):
        load_done(str(p), 2)


def test_pending_items_drops_done():
    pending = _single_func("pending_items", {})
    items = [("g", 0, "x"), ("g", 1, "x"), ("g", 2, "x")]
    assert pending(items, {("g", 0), ("g", 2)}) == [("g", 1, "x")]


def test_audio_done_checks_every_config(tmp_path):
    ns = {"os": os, "_wav_path": None}
    # inject the real _wav_path (pure) so audio_done builds correct names
    ns["_wav_path"] = _single_func("_wav_path", {"os": os})
    audio_done = _single_func("audio_done", ns)
    specs = [(4, 4, 24), (4, 2, 24)]
    for kb, vb, rw in specs:
        (
            tmp_path / f"qwen_g_0_sampling_s0_t0.9_K{kb}_V{vb}_rw={rw}_pl=2.wav"
        ).write_bytes(b"x")
    assert audio_done(str(tmp_path), "g", 0, 0, 0.9, specs, 2) is True
    # same wavs under a different protected_layers do NOT count as done
    assert audio_done(str(tmp_path), "g", 0, 0, 0.9, specs, 0) is False
    (tmp_path / "qwen_g_0_sampling_s0_t0.9_K4_V2_rw=24_pl=2.wav").unlink()
    assert audio_done(str(tmp_path), "g", 0, 0, 0.9, specs, 2) is False


def test_wav_path_encodes_protected_layers(tmp_path):
    wav_path = _single_func("_wav_path", {"os": os})
    p2 = wav_path(str(tmp_path), "g", 0, 0, 0.9, 4, 4, 24, 2)
    p0 = wav_path(str(tmp_path), "g", 0, 0, 0.9, 4, 4, 24, 0)
    assert p2.endswith("_pl=2.wav") and p0.endswith("_pl=0.wav") and p2 != p0


def test_make_patch_records_only_when_active_and_counts_errors():
    ns = _block()

    class _Rec:
        def __init__(self):
            self.active = False
            self.errors = 0
            self.first_error = None
            self.calls = 0
            self.boom = None

        def record(self, *a):
            self.calls += 1
            if self.boom:
                raise RuntimeError(self.boom)

    def original(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
        return "OUT", "ATTN"

    rec = _Rec()
    patched = ns["make_patch"](original, rec)
    assert patched(None, 1, 2, 3, None, 1.0) == ("OUT", "ATTN") and rec.calls == 0
    rec.active = True
    patched(None, 1, 2, 3, None, 1.0)
    assert rec.calls == 1 and rec.errors == 0 and rec.first_error is None
    rec.boom = "boom-1"
    assert patched(None, 1, 2, 3, None, 1.0) == ("OUT", "ATTN") and rec.errors == 1
    # first traceback is captured and kept — later errors only bump the count
    assert "boom-1" in rec.first_error
    rec.boom = "boom-2"
    patched(None, 1, 2, 3, None, 1.0)
    assert rec.errors == 2 and "boom-1" in rec.first_error
