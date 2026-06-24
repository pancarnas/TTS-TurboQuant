"""Shared utilities for rigorous, reproducible TTS-TurboQuant benchmarks.

Lives in the installed ``turboquant`` package so both the Qwen and VALL-E
benchmark scripts can import it without sys.path hacks. Provides:

  - ``set_global_seed`` — seed every RNG that affects token sampling, so a
    baseline run and every compressed config draw the *same* random stream for
    a given (sentence, seed). This is the control that turns config-vs-baseline
    into a paired comparison: any divergence is then attributable to
    compression, not to which random seed each run happened to get.
  - decode-arm helpers — map ``--decode {greedy,sampling,both}`` to the
    generation-kwarg overrides each arm needs.
  - the per-trial CSV schema — one row per (arm, sentence, seed, config), the
    source of truth all downstream statistics are computed from.
"""

from __future__ import annotations

import hashlib
import os
import random
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Decode arms
# ---------------------------------------------------------------------------

VALID_ARMS = ("greedy", "sampling")

# Greedy decode must silence BOTH the talker and the subtalker samplers, else
# the codec codes are still drawn stochastically and the run is not
# deterministic. temperature/top_k are neutralised so they cannot reintroduce
# randomness through generate_config.json defaults.
GREEDY_OVERRIDES: dict[str, Any] = {
    "do_sample": False,
    "subtalker_dosample": False,
    "temperature": 1.0,
    "top_k": 0,
}


def decode_overrides(arm: str, temperature: Optional[float] = None) -> dict[str, Any]:
    """Generation-kwarg overrides for a decode arm.

    ``greedy`` forces deterministic argmax decode; ``sampling`` returns ``{}``
    so the model's configured sampling defaults (temperature, top_k, ...) apply.

    ``temperature`` (sampling arm only) overrides both the talker and subtalker
    sampling temperatures so the whole decode runs at the swept value. Greedy
    ignores it (``do_sample=False`` makes temperature irrelevant), so a swept
    temperature never changes greedy output.
    """
    if arm == "greedy":
        return dict(GREEDY_OVERRIDES)
    if arm == "sampling":
        if temperature is None:
            return {}
        return {"temperature": temperature, "subtalker_temperature": temperature}
    raise ValueError(f"unknown decode arm: {arm!r} (valid: {VALID_ARMS})")


def parse_arms(decode: str) -> list[str]:
    """Map ``--decode {sampling,greedy,both}`` to the ordered arms to run."""
    if decode == "both":
        return ["greedy", "sampling"]
    if decode in VALID_ARMS:
        return [decode]
    raise ValueError(f"unknown --decode value: {decode!r} (valid: both, {VALID_ARMS})")


def parse_seeds(seeds: str) -> list[int]:
    """Parse a comma-separated seed list like ``'0,1,2,3,4'`` into ints."""
    out = [int(s.strip()) for s in seeds.split(",") if s.strip() != ""]
    if not out:
        raise ValueError("no seeds parsed from --seeds")
    return out


def parse_temperatures(temperatures: str) -> list[float]:
    """Parse a comma-separated temperature list like ``'0.7,0.9,1.2'`` into floats.

    Rejects empties and non-positive values (temperature <= 0 is not a valid
    sampling temperature). Used for the sampling-arm temperature sweep.
    """
    out = [float(t.strip()) for t in temperatures.split(",") if t.strip() != ""]
    if not out:
        raise ValueError("no temperatures parsed from --temperatures")
    if any(t <= 0 for t in out):
        raise ValueError("temperatures must be positive")
    return out


# ---------------------------------------------------------------------------
# Deterministic seeding
# ---------------------------------------------------------------------------


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """Seed all RNGs that affect token sampling, for paired comparisons.

    Call immediately before each generation so baseline and every compressed
    config consume the same random stream for a given (sentence, seed). The
    sequences will still diverge once compression flips a sampled token — that
    divergence is the compression signal we want to measure; identical seeding
    removes seed-choice as a confound.

    ``deterministic=True`` additionally requests deterministic CUDA kernels. It
    must take effect before the first CUDA op of the process, so pass it on a
    fresh run; it can slow decode and only warns (not errors) on kernels with no
    deterministic implementation.
    """
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Per-trial CSV schema (source of truth for downstream analysis)
# ---------------------------------------------------------------------------

TRIAL_COLUMNS = [
    "arm",
    "seed",
    "temperature",
    "group",
    "idx",
    "sentence_hash",
    "config",
    "key_bits",
    "value_bits",
    "residual_window",
    "rtf",
    "cer",
    "transcript_len",
    "spk_sim",
    "spk_sim_ref",
    "peak_vram_mb",
    "tokens_per_sec",
    "n_ar_tokens",
    "sim_compressed_mb",
    "realized_mb",
    "theoretical_ratio",
    "effective_ratio",
]


def sentence_hash(text: str) -> str:
    """Stable 8-char hash of a sentence, robust to test-set reordering/edits."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def config_bits(tq_config: Any) -> tuple[Optional[int], Optional[int], Optional[int]]:
    """``(key_bits, value_bits, residual_window)`` for a TurboQuantConfig.

    Returns ``(None, None, None)`` for the baseline (``tq_config is None``) so
    analysis never has to parse the human-readable config string.
    """
    if tq_config is None:
        return None, None, None
    return (
        getattr(tq_config, "key_bits", None),
        getattr(tq_config, "value_bits", None),
        getattr(tq_config, "residual_window", None),
    )


def format_trial_row(values: dict[str, Any]) -> str:
    """Render one CSV row in ``TRIAL_COLUMNS`` order; ``None`` -> empty field."""
    row = []
    for col in TRIAL_COLUMNS:
        v = values.get(col)
        if v is None:
            row.append("")
        elif isinstance(v, float):
            row.append(f"{v:.6g}")
        else:
            row.append(str(v))
    return ",".join(row)
