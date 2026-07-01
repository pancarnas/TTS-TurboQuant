"""Tests for kv_attn_divergence_experiment.

The pure helpers (js_divergence, summarize) and the measurement core
(compressed_attention, DivergenceRecorder, make_patch) are exec-extracted from the
source with their dependencies injected, so the heavy model imports (qwen_tts,
Qwen3TTSModel) are never triggered and everything runs on CPU.
"""

from __future__ import annotations

import os
import types

import pandas as pd
import torch

from turboquant.compressors_v3 import TurboQuantV3

_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "Qwen3-TTS",
    "benchmarks",
    "kv_attn_divergence_experiment.py",
)
_SRC = open(_PATH, encoding="utf-8").read()


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


def _block():
    """Exec js_divergence..make_patch with deps injected; return the namespace."""
    start = _SRC.index("def js_divergence(")
    end = _SRC.index("\ndef force_eager(")
    ns = {
        "torch": torch,
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


# --- pure helpers ---------------------------------------------------------


def test_js_divergence_identical_is_zero():
    js = _single_func("js_divergence", {"torch": torch})
    p = torch.tensor([[0.2, 0.3, 0.5]])
    assert js(p, p.clone()) < 1e-6


def test_js_divergence_disjoint_is_one():
    js = _single_func("js_divergence", {"torch": torch})
    p = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    q = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    assert abs(js(p, q) - 1.0) < 1e-3


def test_js_divergence_symmetric_and_bounded():
    js = _single_func("js_divergence", {"torch": torch})
    p = torch.tensor([[0.7, 0.2, 0.1]])
    q = torch.tensor([[0.1, 0.3, 0.6]])
    assert abs(js(p, q) - js(q, p)) < 1e-6
    assert 0.0 <= js(p, q) <= 1.0


def test_total_variation_bounds():
    tv = _single_func("total_variation", {"torch": torch})
    p = torch.tensor([[0.2, 0.3, 0.5]])
    assert tv(p, p.clone()) < 1e-6
    a = torch.tensor([[1.0, 0.0]])
    b = torch.tensor([[0.0, 1.0]])
    assert abs(tv(a, b) - 1.0) < 1e-6  # disjoint -> all mass moved


def test_top1_agreement():
    top1 = _single_func("top1_agreement", {"torch": torch})
    p = torch.tensor([[0.1, 0.7, 0.2], [0.6, 0.3, 0.1]])
    same = torch.tensor([[0.2, 0.6, 0.2], [0.5, 0.4, 0.1]])  # same argmaxes
    flip = torch.tensor([[0.7, 0.1, 0.2], [0.1, 0.1, 0.8]])  # both argmaxes differ
    assert top1(p, same) == 1.0
    assert top1(p, flip) == 0.0


def test_entropy_delta_sign():
    de = _single_func("entropy_delta", {"torch": torch})
    peaked = torch.tensor([[0.97, 0.01, 0.01, 0.01]])
    uniform = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
    # q wider than p -> positive; q narrower -> negative
    assert de(peaked, uniform) > 0
    assert de(uniform, peaked) < 0


def test_output_cosine():
    oc = _single_func("output_cosine", {"torch": torch})
    v = torch.randn(1, 2, 1, 8)
    assert abs(oc(v, v.clone()) - 1.0) < 1e-6
    assert abs(oc(v, -v) + 1.0) < 1e-6  # opposite -> -1


def test_force_eager_walks_wrapped_model():
    # Qwen3TTSModel is a plain wrapper (no .modules()); the talker is at .model.
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
    force_eager(w)  # must not raise, and must reach the nested attention config
    assert w.model.attn.config._attn_implementation == "eager"


_METRIC_NAMES = [
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


def test_streaming_aggregate_matches_means_and_diff():
    acc = _single_func("accumulate_running", {"_METRICS": _METRIC_NAMES})
    frame = _single_func("running_frame", {"_METRICS": _METRIC_NAMES, "pd": pd})

    def _row(rw, v):  # (group, idx, layer, pos, rw, *9 metrics)
        return ("g", 0, 0, 10, rw) + tuple([v] * 9)

    running: dict = {}
    # stream in two batches (simulating per-sentence flushes) then aggregate
    acc(running, [_row(24, 0.01), _row(0, 0.20)])
    acc(running, [_row(24, 0.03), _row(0, 0.30)])
    assert running[24]["_n"] == 2 and running[0]["_n"] == 2
    out = frame(running)
    assert list(out.index)[:2] == [24, 0]  # high -> low
    assert abs(out.loc[24, "attn_js"] - 0.02) < 1e-9
    assert abs(out.loc[0, "attn_js"] - 0.25) < 1e-9
    assert abs(out.loc["diff(0-24)", "attn_js"] - 0.23) < 1e-9


def test_summarize_diff_row_and_order():
    summarize = _single_func("summarize", {"pd": pd})
    df = pd.DataFrame(
        {
            "rw": [24, 24, 0, 0],
            "attn_js": [0.01, 0.03, 0.20, 0.30],
            "attn_tv": [0.01, 0.03, 0.20, 0.30],
            "attn_top1": [0.99, 0.99, 0.80, 0.80],
            "attn_dentropy": [0.01, 0.01, 0.10, 0.10],
            "out_cos": [0.99, 0.99, 0.90, 0.90],
            "cos_k": [0.99, 0.99, 0.90, 0.90],
            "cos_v": [0.99, 0.99, 0.92, 0.92],
            "relmse_k": [0.01, 0.01, 0.10, 0.10],
            "relmse_v": [0.01, 0.01, 0.08, 0.08],
        }
    )
    out = summarize(df)
    assert list(out.index)[:2] == [24, 0]
    assert abs(out.loc[24, "attn_js"] - 0.02) < 1e-9
    assert abs(out.loc[0, "attn_js"] - 0.25) < 1e-9
    assert abs(out.loc["diff(0-24)", "attn_js"] - 0.23) < 1e-9


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
    assert aw.shape == (1, nkv_x_groups := groups * 2, 1, 32)
    assert torch.allclose(aw.sum(-1), torch.ones_like(aw.sum(-1)), atol=1e-4)


def test_recorder_logs_one_row_per_rw_with_sane_values():
    ns = _block()
    q, key, value, hd, groups = _attn_inputs(64)
    scaling = hd**-0.5
    rec = ns["DivergenceRecorder"](
        rws=[16, 0], kb=4, vb=4, n_layers=4, prot_layers=0, prot_bits=8, stride=1
    )
    rec.group, rec.idx = "g", 0
    af = ns["compressed_attention"](q, key, groups, None, scaling)  # fp16 attention
    rec.record(_module(groups), q, key, value, None, scaling, af)

    assert rec.errors == 0
    assert len(rec.rows) == 2  # one per rw
    # map row by name (record() appends in this column order)
    cols = [
        "group",
        "idx",
        "layer",
        "pos",
        "rw",
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
    by_rw = {dict(zip(cols, r))["rw"]: dict(zip(cols, r)) for r in rec.rows}
    for d in by_rw.values():
        assert d["pos"] == 64
        assert 0.0 <= d["attn_js"] <= 1.0 + 1e-6
        assert 0.0 <= d["attn_tv"] <= 1.0 + 1e-6
        assert 0.0 <= d["attn_top1"] <= 1.0 + 1e-6
        assert d["cos_k"] <= 1.0 + 1e-6 and d["out_cos"] <= 1.0 + 1e-6
        assert d["relmse_k"] >= 0.0 and d["relmse_v"] >= 0.0
    # rw=0 quantizes ALL tokens; rw=16 keeps the last 16 exact -> rw=0 distorts more
    assert by_rw[0]["relmse_k"] >= by_rw[16]["relmse_k"]
    assert by_rw[0]["attn_js"] >= by_rw[16]["attn_js"]
    assert by_rw[0]["attn_tv"] >= by_rw[16]["attn_tv"]
    assert by_rw[0]["cos_k"] <= by_rw[16]["cos_k"]  # lower = worse
    assert by_rw[0]["out_cos"] <= by_rw[16]["out_cos"]  # output drifts more
    assert by_rw[0]["attn_top1"] <= by_rw[16]["attn_top1"]  # more argmax flips


def test_recorder_step_stride_skips_unrecorded_positions():
    ns = _block()
    rec = ns["DivergenceRecorder"](
        rws=[0], kb=4, vb=4, n_layers=4, prot_layers=0, prot_bits=8, stride=4
    )
    rec.group, rec.idx = "g", 0
    scaling = 64**-0.5
    # seq=63 -> pos 63 not divisible by 4 -> skipped
    q, key, value, hd, groups = _attn_inputs(63)
    af = ns["compressed_attention"](q, key, groups, None, scaling)
    rec.record(_module(groups), q, key, value, None, scaling, af)
    assert rec.rows == []
    # seq=64 -> pos 64 divisible by 4 -> recorded
    q, key, value, hd, groups = _attn_inputs(64)
    af = ns["compressed_attention"](q, key, groups, None, scaling)
    rec.record(_module(groups), q, key, value, None, scaling, af)
    assert len(rec.rows) == 1


def test_recorder_reuses_compressors_across_calls():
    ns = _block()
    rec = ns["DivergenceRecorder"](
        rws=[16, 0], kb=4, vb=4, n_layers=4, prot_layers=0, prot_bits=8, stride=1
    )
    rec.group, rec.idx = "g", 0
    scaling = 64**-0.5
    for _ in range(3):
        q, key, value, hd, groups = _attn_inputs(64)
        af = ns["compressed_attention"](q, key, groups, None, scaling)
        rec.record(_module(groups), q, key, value, None, scaling, af)
    assert len(rec._cache) == 2  # one TurboQuantV3 per (layer, rw), reused


def test_make_patch_records_only_when_active_and_counts_errors():
    ns = _block()

    class _Rec:
        def __init__(self):
            self.active = False
            self.errors = 0
            self.calls = 0

        def record(self, *a):
            self.calls += 1
            if self.boom:
                raise RuntimeError("boom")

        boom = False

    def original(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
        return "OUT", "ATTN"

    rec = _Rec()
    patched = ns["make_patch"](original, rec)
    # inactive: passes through, no record
    assert patched(None, 1, 2, 3, None, 1.0) == ("OUT", "ATTN")
    assert rec.calls == 0
    # active: records
    rec.active = True
    patched(None, 1, 2, 3, None, 1.0)
    assert rec.calls == 1 and rec.errors == 0
    # record raises -> error counted, output still returned
    rec.boom = True
    assert patched(None, 1, 2, 3, None, 1.0) == ("OUT", "ATTN")
    assert rec.errors == 1
