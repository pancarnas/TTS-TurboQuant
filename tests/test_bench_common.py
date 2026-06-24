"""Tests for the shared benchmark utilities (pure functions only — no GPU)."""

import pytest

from turboquant.bench_common import (
    TRIAL_COLUMNS,
    config_bits,
    decode_overrides,
    format_trial_row,
    parse_arms,
    parse_seeds,
    parse_temperatures,
    sentence_hash,
    set_global_seed,
)
from turboquant.config import TurboQuantConfig


def test_parse_seeds_basic() -> None:
    assert parse_seeds("0,1,2,3,4") == [0, 1, 2, 3, 4]


def test_parse_seeds_tolerates_whitespace_and_trailing_comma() -> None:
    assert parse_seeds(" 7, 8 ,9, ") == [7, 8, 9]


def test_parse_seeds_rejects_empty() -> None:
    with pytest.raises(ValueError):
        parse_seeds("")


def test_parse_arms_both_is_greedy_then_sampling() -> None:
    assert parse_arms("both") == ["greedy", "sampling"]


def test_parse_arms_single() -> None:
    assert parse_arms("greedy") == ["greedy"]
    assert parse_arms("sampling") == ["sampling"]


def test_parse_arms_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        parse_arms("beam")


def test_parse_temperatures_basic() -> None:
    assert parse_temperatures("0.7,0.9,1.2") == [0.7, 0.9, 1.2]


def test_parse_temperatures_tolerates_whitespace_and_trailing_comma() -> None:
    assert parse_temperatures(" 0.8 , 1.0 ,1.5, ") == [0.8, 1.0, 1.5]


def test_parse_temperatures_rejects_empty() -> None:
    with pytest.raises(ValueError):
        parse_temperatures("")


def test_parse_temperatures_rejects_non_positive() -> None:
    with pytest.raises(ValueError):
        parse_temperatures("0.9,0,1.2")
    with pytest.raises(ValueError):
        parse_temperatures("-0.5")


def test_decode_overrides_sampling_temperature_sets_both_talker_and_subtalker() -> None:
    ov = decode_overrides("sampling", temperature=1.2)
    assert ov["temperature"] == 1.2
    assert ov["subtalker_temperature"] == 1.2


def test_decode_overrides_sampling_without_temperature_is_unchanged() -> None:
    assert decode_overrides("sampling") == {}
    assert decode_overrides("sampling", temperature=None) == {}


def test_decode_overrides_greedy_ignores_temperature() -> None:
    # Greedy does not sample, so a swept temperature must not change it.
    assert decode_overrides("greedy", temperature=1.2) == decode_overrides("greedy")


def test_temperature_is_in_trial_schema() -> None:
    assert "temperature" in TRIAL_COLUMNS


def test_spk_sim_ref_is_in_trial_schema() -> None:
    # Ground-truth speaker similarity reported side-by-side with baseline-as-ref.
    assert "spk_sim_ref" in TRIAL_COLUMNS
    assert TRIAL_COLUMNS.index("spk_sim_ref") == TRIAL_COLUMNS.index("spk_sim") + 1


def test_decode_overrides_greedy_forces_both_samplers_off() -> None:
    ov = decode_overrides("greedy")
    assert ov["do_sample"] is False
    assert ov["subtalker_dosample"] is False


def test_decode_overrides_sampling_is_empty() -> None:
    assert decode_overrides("sampling") == {}


def test_decode_overrides_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        decode_overrides("nucleus")


def test_config_bits_baseline_is_none() -> None:
    assert config_bits(None) == (None, None, None)


def test_config_bits_extracts_fields() -> None:
    cfg = TurboQuantConfig(key_bits=4, value_bits=2, residual_window=128)
    assert config_bits(cfg) == (4, 2, 128)


def test_sentence_hash_is_stable_and_short() -> None:
    h1 = sentence_hash("Hello, how are you doing today?")
    h2 = sentence_hash("Hello, how are you doing today?")
    assert h1 == h2
    assert len(h1) == 8
    assert h1 != sentence_hash("A different sentence.")


def test_format_trial_row_orders_by_schema_and_blanks_none() -> None:
    row = format_trial_row(
        {
            "arm": "greedy",
            "seed": 0,
            "group": "short",
            "idx": 1,
            "config": "K4/V2",
            "cer": 0.0123,
            "spk_sim": None,
        }
    )
    fields = row.split(",")
    assert len(fields) == len(TRIAL_COLUMNS)
    # index by column name so the assertion survives schema additions
    assert fields[TRIAL_COLUMNS.index("arm")] == "greedy"
    assert fields[TRIAL_COLUMNS.index("seed")] == "0"
    assert fields[TRIAL_COLUMNS.index("group")] == "short"
    assert fields[TRIAL_COLUMNS.index("idx")] == "1"
    # unset temperature -> empty field
    assert fields[TRIAL_COLUMNS.index("temperature")] == ""
    # spk_sim is None -> empty field
    assert fields[TRIAL_COLUMNS.index("spk_sim")] == ""
    # float rendered with %g
    assert fields[TRIAL_COLUMNS.index("cer")] == "0.0123"


def test_set_global_seed_makes_torch_rng_reproducible() -> None:
    torch = pytest.importorskip("torch")
    set_global_seed(123)
    a = torch.randn(5)
    set_global_seed(123)
    b = torch.randn(5)
    assert torch.equal(a, b)
