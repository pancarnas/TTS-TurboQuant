"""GPU-free test for the kv_recon_experiment aggregation (summarize)."""

from __future__ import annotations

import os

import pandas as pd

_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "Qwen3-TTS",
    "benchmarks",
    "kv_recon_experiment.py",
)


def _load_summarize():
    """Import only ``summarize`` without triggering the heavy model imports.

    The module's top-level imports (qwen_tts, torch model) aren't available in a
    plain test env, so we exec the source and grab the pure function. ``summarize``
    has no module-level deps beyond pandas, so this is safe.
    """
    src = open(_PATH, encoding="utf-8").read()
    # Keep only the summarize function body to avoid importing the model stack.
    start = src.index("def summarize(")
    end = src.index("\ndef ", start + 1)
    namespace: dict = {"pd": pd}
    exec(compile(src[start:end], _PATH, "exec"), namespace)
    return namespace["summarize"]


def _df(rows):
    return pd.DataFrame(
        rows,
        columns=["residual_window", "cos_k", "cos_v", "relmse_k", "relmse_v"],
    )


def test_summarize_means_and_diff_row():
    summarize = _load_summarize()
    df = _df(
        [
            (24, 0.99, 0.98, 0.01, 0.02),
            (24, 0.97, 0.96, 0.03, 0.04),  # rw24 mean cos_k = 0.98
            (0, 0.90, 0.88, 0.10, 0.12),
            (0, 0.80, 0.84, 0.20, 0.16),  # rw0 mean cos_k = 0.85
        ]
    )
    out = summarize(df)
    # rows ordered high→low rw, plus a diff row
    assert 24 in out.index and 0 in out.index
    assert abs(out.loc[24, "cos_k"] - 0.98) < 1e-9
    assert abs(out.loc[0, "cos_k"] - 0.85) < 1e-9
    diff = out.loc["diff(0-24)"]
    assert abs(diff["cos_k"] - (0.85 - 0.98)) < 1e-9  # rw0 worse → negative
    assert abs(diff["relmse_k"] - (0.15 - 0.02)) < 1e-9  # rw0 higher error → positive


def test_summarize_no_diff_when_single_window():
    summarize = _load_summarize()
    out = summarize(_df([(24, 0.99, 0.98, 0.01, 0.02)]))
    assert list(out.index) == [24]
    assert not any(str(i).startswith("diff") for i in out.index)
