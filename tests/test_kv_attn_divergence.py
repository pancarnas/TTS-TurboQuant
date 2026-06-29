"""GPU-free tests for kv_attn_divergence_experiment pure functions.

js_divergence (torch) and summarize (pandas) are exec-extracted from the source so
the heavy model imports (qwen_tts, Qwen3TTSModel) are never triggered.
"""

from __future__ import annotations

import os

import pandas as pd
import torch

_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "Qwen3-TTS",
    "benchmarks",
    "kv_attn_divergence_experiment.py",
)


def _extract(func_name: str, namespace: dict):
    src = open(_PATH, encoding="utf-8").read()
    start = src.index(f"def {func_name}(")
    end = src.index("\ndef ", start + 1)
    exec(compile(src[start:end], _PATH, "exec"), namespace)
    return namespace[func_name]


def test_js_divergence_identical_is_zero():
    js = _extract("js_divergence", {"torch": torch})
    p = torch.tensor([[0.2, 0.3, 0.5]])
    assert js(p, p.clone()) < 1e-6


def test_js_divergence_disjoint_is_one():
    js = _extract("js_divergence", {"torch": torch})
    p = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    q = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    assert abs(js(p, q) - 1.0) < 1e-3  # base-2 JS of disjoint dists -> 1


def test_js_divergence_symmetric_and_bounded():
    js = _extract("js_divergence", {"torch": torch})
    p = torch.tensor([[0.7, 0.2, 0.1]])
    q = torch.tensor([[0.1, 0.3, 0.6]])
    a, b = js(p, q), js(q, p)
    assert abs(a - b) < 1e-6
    assert 0.0 <= a <= 1.0


def test_summarize_adds_diff_row_high_to_low():
    summarize = _extract("summarize", {"pd": pd})
    df = pd.DataFrame(
        {
            "rw": [24, 24, 0, 0],
            "attn_js": [0.01, 0.03, 0.20, 0.30],  # rw24 mean .02, rw0 mean .25
            "cos_k": [0.99, 0.99, 0.90, 0.90],
            "cos_v": [0.99, 0.99, 0.92, 0.92],
            "relmse_k": [0.01, 0.01, 0.10, 0.10],
            "relmse_v": [0.01, 0.01, 0.08, 0.08],
        }
    )
    out = summarize(df)
    assert list(out.index)[:2] == [24, 0]  # high -> low
    assert abs(out.loc[24, "attn_js"] - 0.02) < 1e-9
    assert abs(out.loc[0, "attn_js"] - 0.25) < 1e-9
    # rw0 moves attention more -> positive diff
    assert abs(out.loc["diff(0-24)", "attn_js"] - (0.25 - 0.02)) < 1e-9


def test_summarize_no_diff_single_window():
    summarize = _extract("summarize", {"pd": pd})
    df = pd.DataFrame(
        {
            "rw": [24],
            "attn_js": [0.02],
            "cos_k": [0.99],
            "cos_v": [0.99],
            "relmse_k": [0.01],
            "relmse_v": [0.01],
        }
    )
    out = summarize(df)
    assert list(out.index) == [24]
