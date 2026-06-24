"""Tests for the shared evaluation-sentence module (pure — no GPU, no network).

Pins the contract both benchmark scripts depend on: the EvalItem schema, the
.lst / ELLA-V parsers, the group iterator (slicing + ordering + unknown-group
error), and the shard partitioner that lets several workers split the work-list
without dropping or duplicating a cell.
"""

import pytest

from turboquant.eval_sentences import (
    EvalItem,
    as_eval_item,
    iter_eval_items,
    load_ellav_hard,
    parse_seedtts_lst,
    shard_cells,
)


# --- EvalItem schema --------------------------------------------------------


def test_eval_item_is_frozen_with_optional_reference_fields() -> None:
    item = EvalItem(text="Hello there.", group="smoke")
    assert item.text == "Hello there."
    assert item.group == "smoke"
    assert item.ref_audio is None
    assert item.ref_text is None
    assert item.ground_truth_audio is None
    with pytest.raises(Exception):  # frozen dataclass → assignment forbidden
        item.text = "mutated"  # type: ignore[misc]


def test_as_eval_item_wraps_a_bare_string_for_back_compat() -> None:
    item = as_eval_item("A plain sentence.", "long")
    assert isinstance(item, EvalItem)
    assert item.text == "A plain sentence."
    assert item.group == "long"
    # an EvalItem passes through unchanged (group preserved)
    existing = EvalItem(text="x", group="ellav_hard")
    assert as_eval_item(existing, "ignored") is existing


# --- seed-tts-eval .lst parser ---------------------------------------------


def test_parse_seedtts_lst_four_fields_has_no_ground_truth() -> None:
    line = "cv_en_1|We asked over twenty people.|prompt-wavs/cv_en_1.wav|Get to the bank early."
    items = parse_seedtts_lst([line], audio_root="/data/seedtts", group="seedtts_en")
    assert len(items) == 1
    it = items[0]
    assert it.text == "Get to the bank early."
    assert it.ref_text == "We asked over twenty people."
    assert it.ref_audio == "/data/seedtts/prompt-wavs/cv_en_1.wav"
    assert it.ground_truth_audio is None
    assert it.group == "seedtts_en"


def test_parse_seedtts_lst_five_fields_carries_ground_truth() -> None:
    line = "id2|prompt text|prompt-wavs/p2.wav|target text|wavs/gt2.wav"
    items = parse_seedtts_lst([line], audio_root="/data/seedtts", group="seedtts_en")
    assert items[0].ground_truth_audio == "/data/seedtts/wavs/gt2.wav"


def test_parse_seedtts_lst_skips_blank_and_malformed_lines() -> None:
    lines = ["", "   ", "only|two", "a|b|c|d"]
    items = parse_seedtts_lst(lines, audio_root="/r", group="seedtts_en")
    assert len(items) == 1  # only the well-formed 4-field row survives


# --- ELLA-V hard text loader ------------------------------------------------


def test_load_ellav_hard_one_sentence_per_line(tmp_path) -> None:
    p = tmp_path / "ellav_hard.txt"
    p.write_text(
        "# a comment header\n"
        "the great greek grape growers grow great greek grapes\n"
        "\n"
        "How many cookies could a good cook cook?\n",
        encoding="utf-8",
    )
    items = load_ellav_hard(str(p))
    assert [i.text for i in items] == [
        "the great greek grape growers grow great greek grapes",
        "How many cookies could a good cook cook?",
    ]
    assert all(i.group == "ellav_hard" for i in items)
    assert all(i.ref_audio is None for i in items)  # text-only set


# --- group iterator ---------------------------------------------------------


def test_iter_eval_items_curated_group_is_nonempty_and_typed() -> None:
    items = iter_eval_items(["smoke"], max_per_group=None, data_dir=None)
    assert items, "smoke group should ship curated sentences"
    assert all(isinstance(i, EvalItem) and i.group == "smoke" for i in items)


def test_iter_eval_items_max_per_group_slices_each_group() -> None:
    one = iter_eval_items(["smoke", "long"], max_per_group=1, data_dir=None)
    groups = [i.group for i in one]
    assert groups.count("smoke") == 1
    assert groups.count("long") == 1


def test_iter_eval_items_is_deterministic() -> None:
    a = iter_eval_items(["smoke", "long"], max_per_group=None, data_dir=None)
    b = iter_eval_items(["smoke", "long"], max_per_group=None, data_dir=None)
    assert [i.text for i in a] == [i.text for i in b]


def test_iter_eval_items_rejects_unknown_group() -> None:
    with pytest.raises(ValueError):
        iter_eval_items(["does_not_exist"], max_per_group=None, data_dir=None)


def test_long_group_has_a_spread_of_lengths() -> None:
    """The long-context set must actually vary in length to stress the cache."""
    longs = iter_eval_items(["long"], max_per_group=None, data_dir=None)
    word_counts = sorted(len(i.text.split()) for i in longs)
    assert len(word_counts) >= 3
    assert word_counts[-1] >= 2 * word_counts[0]  # genuine spread


# --- shard partitioner ------------------------------------------------------


def test_shard_cells_partitions_without_loss_or_duplication() -> None:
    cells = list(range(23))
    num_shards = 4
    shards = [shard_cells(cells, num_shards, i) for i in range(num_shards)]
    # disjoint
    seen: set[int] = set()
    for s in shards:
        assert seen.isdisjoint(s)
        seen.update(s)
    # complete
    assert seen == set(cells)


def test_shard_cells_single_shard_is_identity() -> None:
    cells = list(range(10))
    assert shard_cells(cells, 1, 0) == cells


def test_shard_cells_rejects_bad_shard_id() -> None:
    with pytest.raises(ValueError):
        shard_cells([1, 2, 3], num_shards=2, shard_id=2)
