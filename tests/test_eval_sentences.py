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
    available_groups,
    iter_eval_items,
    length_category,
    load_curated_cell,
    load_ellav_hard,
    load_librispeech_pc,
    load_libritts_long,
    load_seedtts_cell,
    parse_seedtts_lst,
    plan_long_concatenations,
    predict_tokens,
    shard_cells,
    text_difficulty,
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


# --- length heuristic + difficulty classifier (3×3 grid) --------------------


def test_predict_tokens_uses_word_count_heuristic() -> None:
    # ≈7 tokens/word (conservative word→talker-token estimate)
    assert predict_tokens("one two three four five") == round(7.0 * 5)
    assert predict_tokens("") == 0


def test_length_category_bins_by_predicted_tokens() -> None:
    assert length_category("hello there friend") == "short"  # 3 words → ~19 tok
    medium = " ".join(["word"] * 30)  # ~192 tok
    assert length_category(medium) == "medium"
    long = " ".join(["word"] * 120)  # ~768 tok
    assert length_category(long) == "long"


def test_text_difficulty_easy_medium_hard() -> None:
    assert text_difficulty("the cat sat on the warm mat today") == "easy"
    # proper noun + number → medium
    assert text_difficulty("Sarah bought 12 apples at the market") == "medium"
    # adjacent repetition → hard
    assert text_difficulty("she said said the words very very fast") == "hard"
    # alliteration run → hard
    assert text_difficulty("Peter picked a peck of pickled peppers proudly") == "hard"


def test_grid_cells_are_available_groups() -> None:
    groups = available_groups()
    for length in ("short", "medium", "long"):
        for diff in ("easy", "medium", "hard"):
            assert f"{length}_{diff}" in groups


def test_load_curated_cell_reads_data_dir(tmp_path) -> None:
    cell_dir = tmp_path / "curated"
    cell_dir.mkdir()
    (cell_dir / "long_hard.txt").write_text(
        "# header comment\nfirst dense passage here\n\nsecond passage too\n",
        encoding="utf-8",
    )
    items = load_curated_cell(str(tmp_path), "long_hard")
    assert [i.text for i in items] == [
        "first dense passage here",
        "second passage too",
    ]
    assert all(i.group == "long_hard" and i.ref_audio is None for i in items)


def test_load_seedtts_cell_filters_and_retags(tmp_path) -> None:
    en = tmp_path / "en"
    en.mkdir()
    (en / "prompt-wavs").mkdir()
    # short+easy natural line vs a medium (has number+proper noun) line
    (en / "meta.lst").write_text(
        "id1|prompt|prompt-wavs/p1.wav|the dog ran across the green field\n"
        "id2|prompt|prompt-wavs/p2.wav|Sarah counted 15 red apples in March\n",
        encoding="utf-8",
    )
    easy = load_seedtts_cell(str(tmp_path), "short_easy")
    assert [i.text for i in easy] == ["the dog ran across the green field"]
    assert easy[0].group == "short_easy"
    assert easy[0].ref_audio.endswith("prompt-wavs/p1.wav")  # own ref kept
    medium = load_seedtts_cell(str(tmp_path), "short_medium")
    assert [i.text for i in medium] == ["Sarah counted 15 red apples in March"]


# --- standard-corpus loaders (paper-foundation) -----------------------------


def test_load_librispeech_pc_six_field_tab(tmp_path) -> None:
    root = tmp_path / "librispeech_pc"
    root.mkdir()
    # Real F5-TTS format: ref_utt, ref_dur, ref_text, gen_utt, gen_dur, gen_text (TAB).
    (root / "librispeech_pc_test_clean_cross_sentence.lst").write_text(
        "4992-41806-0009\t4.35\texclaimed Bill to his wife\t"
        "4992-23283-0000\t6.64\tBut the more forgetfulness had then prevailed\n",
        encoding="utf-8",
    )
    items = load_librispeech_pc(str(tmp_path))
    assert len(items) == 1
    it = items[0]
    assert it.group == "librispeech_pc"
    assert it.text == "But the more forgetfulness had then prevailed"  # gen_text
    assert it.ref_text == "exclaimed Bill to his wife"  # ref_text (prompt)
    # ids resolve to flac under LibriSpeech/test-clean/SPK/CHAPTER/ID.flac
    assert it.ref_audio.endswith(
        "LibriSpeech/test-clean/4992/41806/4992-41806-0009.flac"
    )
    assert it.ground_truth_audio.endswith(
        "LibriSpeech/test-clean/4992/23283/4992-23283-0000.flac"
    )


def test_load_libritts_long_parses_manifest(tmp_path) -> None:
    root = tmp_path / "libritts_long"
    root.mkdir()
    (root / "manifest.lst").write_text(
        "long0|prompt text|refs/r0.wav|a long concatenated passage here|gt/g0.wav\n",
        encoding="utf-8",
    )
    items = load_libritts_long(str(tmp_path))
    assert len(items) == 1 and items[0].group == "libritts_long"
    assert items[0].ground_truth_audio.endswith("gt/g0.wav")


def _utt(speaker, chapter, idx, words, wav):
    return {
        "speaker": speaker,
        "chapter": chapter,
        "idx": idx,
        "text": " ".join(["word"] * words),
        "wav": wav,
    }


def test_plan_long_concatenations_reaches_buckets_nonoverlapping() -> None:
    # ~6.4 tok/word → ~16 words ≈ 100 tok per utt; bucket 256 needs ~3 utts.
    utts = [_utt("spkA", "ch1", i, 16, f"a_{i}.wav") for i in range(12)]
    utts += [_utt("spkA", "ch2", 0, 16, "a_ref.wav")]  # other-chapter ref source
    plan = plan_long_concatenations(utts, buckets=(256, 512))
    assert plan, "should emit passages"
    for rec in plan:
        assert rec["actual_tokens"] >= rec["bucket"]  # reached the target
        assert rec["ref_wav"] not in rec["member_wavs"]  # ref not leaked
        assert rec["ref_wav"] is not None  # same-speaker ref assigned
    # Non-overlapping within a bucket: each member wav used at most once per bucket.
    for bucket in (256, 512):
        used = [w for r in plan if r["bucket"] == bucket for w in r["member_wavs"]]
        assert len(used) == len(set(used))


def test_assign_rotating_refs_for_ellav(tmp_path) -> None:
    # ELLA-V (text-only) gets references rotated from a LibriSpeech-PC pool.
    root = tmp_path / "librispeech_pc"
    root.mkdir()
    (root / "librispeech_pc_test_clean_cross_sentence.lst").write_text(
        "10-20-0001\t3.0\tprompt one\t10-20-0002\t3.0\ttarget one\n"
        "11-21-0001\t3.0\tprompt two\t11-21-0002\t3.0\ttarget two\n",
        encoding="utf-8",
    )
    ella_path = tmp_path / "ellav_hard.txt"
    ella_path.write_text(
        "she sells sea shells by the shore\n"
        "the the report report said it twice\n"
        "Peter picked a peck of pickled peppers proudly\n",
        encoding="utf-8",
    )
    items = iter_eval_items(["ellav_hard"], data_dir=str(tmp_path))
    refs = [i.ref_audio for i in items]
    assert all(r is not None for r in refs)  # all got a reference
    assert refs[0].endswith("10/20/10-20-0001.flac")
    assert refs[1].endswith("11/21/11-21-0001.flac")
    assert refs[2].endswith("10/20/10-20-0001.flac")  # rotation wraps (pool of 2)


def test_standard_groups_registered() -> None:
    groups = available_groups()
    assert "librispeech_pc" in groups and "libritts_long" in groups
