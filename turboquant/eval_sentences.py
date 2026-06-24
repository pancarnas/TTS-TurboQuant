"""Shared evaluation-sentence set for both TTS-TurboQuant benchmarks.

Replaces the per-script inline ``SENTENCE_GROUPS`` so Qwen and VALL-E evaluate
the *same* text (A/B parity by construction). A sentence is an :class:`EvalItem`
rather than a bare string, because the SOTA-comparable groups carry a reference
clip (for voice cloning + ground-truth speaker similarity), not just text.

Groups come from two places:
  - **curated literals** (``smoke``, ``long``) — defined here; ``long`` is the
    long-context stress set (graduated lengths) that is this project's unique
    axis, since KV-compression error grows with sequence length.
  - **disk-backed standard sets** (``seedtts_en``, ``ellav_hard``) — loaded from
    ``data/`` after ``tools/fetch_eval_data.py`` downloads them.

The shard helper lets several workers split the deterministic work-list without
dropping or duplicating a cell (see the parallel-execution design).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterable, Optional, Union


@dataclass(frozen=True)
class EvalItem:
    """One evaluation case: target text plus optional reference audio.

    ``ref_audio``/``ref_text`` drive zero-shot voice cloning (None → use the
    model's preset speaker). ``ground_truth_audio`` is a real clip of the target
    text for *ground-truth* speaker similarity (None → skip that metric).
    """

    text: str
    group: str
    ref_audio: Optional[str] = None
    ref_text: Optional[str] = None
    ground_truth_audio: Optional[str] = None


def as_eval_item(value: Union[str, EvalItem], group: str) -> EvalItem:
    """Wrap a bare string as an :class:`EvalItem`; pass an EvalItem through."""
    if isinstance(value, EvalItem):
        return value
    return EvalItem(text=value, group=group)


# ---------------------------------------------------------------------------
# Disk-backed standard sets
# ---------------------------------------------------------------------------


def parse_seedtts_lst(
    lines: Iterable[str], audio_root: str, group: str = "seedtts_en"
) -> list[EvalItem]:
    """Parse seed-tts-eval ``.lst`` lines into EvalItems.

    Each row is ``id | prompt_text | prompt_wav | target_text`` (4 fields) and the
    cross-speaker set adds a 5th ``target_wav`` (ground truth). Relative wav paths
    are resolved against ``audio_root``. Blank/short (<4 field) lines are skipped.
    """
    items: list[EvalItem] = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < 4:
            continue
        _, prompt_text, prompt_wav, target_text = parts[0], parts[1], parts[2], parts[3]
        has_gt = len(parts) >= 5 and parts[4].strip() != ""
        items.append(
            EvalItem(
                text=target_text,
                group=group,
                ref_audio=os.path.join(audio_root, prompt_wav),
                ref_text=prompt_text,
                ground_truth_audio=os.path.join(audio_root, parts[4])
                if has_gt
                else None,
            )
        )
    return items


def load_seedtts_en(
    data_dir: str, limit: Optional[int] = None, cross_speaker: bool = False
) -> list[EvalItem]:
    """Load seed-tts-eval ``test-en`` from ``data_dir`` (see ``fetch_eval_data``).

    Wav paths in the ``.lst`` are relative to the ``.lst``'s own ``en/`` directory
    (the repo stores them under ``en/prompt-wavs`` / ``en/wavs``), so resolve audio
    against ``<data_dir>/en`` when that layout is present, else ``<data_dir>``.
    """
    name = "non_para_reconstruct_meta.lst" if cross_speaker else "meta.lst"
    lst_path = os.path.join(data_dir, "en", name)
    en_dir = os.path.join(data_dir, "en")
    audio_root = (
        en_dir if os.path.isdir(os.path.join(en_dir, "prompt-wavs")) else data_dir
    )
    with open(lst_path, encoding="utf-8") as fh:
        items = parse_seedtts_lst(fh, audio_root=audio_root, group="seedtts_en")
    return items[:limit] if limit else items


def load_ellav_hard(path: str) -> list[EvalItem]:
    """Load the ELLA-V hard sentences (one per line; ``#`` lines are comments)."""
    items: list[EvalItem] = []
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            items.append(EvalItem(text=line, group="ellav_hard"))
    return items


# ---------------------------------------------------------------------------
# Curated literals — smoke + long-context stress set
# ---------------------------------------------------------------------------

_SMOKE = [
    "Hello, how are you doing today?",
    "The weather is beautiful this morning.",
]

_LONG_PASSAGE_A = (
    "The history of human civilization is a remarkable story of innovation and "
    "perseverance. From the earliest cave paintings to the development of written "
    "language, each generation has built upon the achievements of those who came "
    "before. The invention of the wheel, the printing press, and the steam engine "
    "each reshaped how people lived and worked. In the modern era, electricity and "
    "the telephone shrank distances that had once taken weeks to cross. Today we "
    "stand at a crossroads where artificial intelligence and biotechnology promise "
    "to transform our world in ways we can barely imagine. Some welcome these "
    "changes with open arms, while others warn of unintended consequences that may "
    "take generations to fully understand. Whatever path we choose, the decisions "
    "made in the coming decades will echo far into the future, shaping the lives of "
    "people who have not yet been born."
)

_LONG_PASSAGE_B = (
    "The ocean covers more than seventy percent of the surface of our planet and "
    "holds the vast majority of its water. Despite centuries of patient exploration, "
    "we have charted only a small fraction of the sea floor in any real detail. The "
    "deep ocean remains one of the last great frontiers, home to strange creatures "
    "that have evolved in complete darkness, under crushing pressure, and at "
    "temperatures hovering just above freezing. Scientists who study these regions "
    "often compare them to alien worlds, so different are they from the sunlit "
    "shallows we know. Every expedition seems to return with species never seen "
    "before, hinting at how much remains hidden beneath the waves. Protecting these "
    "fragile ecosystems has become an urgent task, for the choices we make on land "
    "ripple outward through currents that touch every shore on Earth."
)


def _truncate_to_words(text: str, max_words: int) -> str:
    """Trim ``text`` to the fewest whole sentences covering ``max_words`` words."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    out: list[str] = []
    count = 0
    for sentence in sentences:
        out.append(sentence)
        count += len(sentence.split())
        if count >= max_words:
            break
    return " ".join(out)


def _long_items() -> list[EvalItem]:
    """Graduated-length long-context items (~50 / ~100 / full / combined words)."""
    combined = f"{_LONG_PASSAGE_A} {_LONG_PASSAGE_B}"
    texts = [
        _truncate_to_words(_LONG_PASSAGE_A, 50),
        _truncate_to_words(_LONG_PASSAGE_A, 100),
        _LONG_PASSAGE_A,
        combined,
    ]
    return [EvalItem(text=t, group="long") for t in texts]


_CURATED_GROUPS: dict[str, list[EvalItem]] = {
    "smoke": [EvalItem(text=t, group="smoke") for t in _SMOKE],
    "long": _long_items(),
}

_DISK_GROUPS = ("seedtts_en", "ellav_hard")


def available_groups() -> list[str]:
    """All valid group names (curated literals + disk-backed standard sets)."""
    return list(_CURATED_GROUPS) + list(_DISK_GROUPS)


def _valid_groups() -> list[str]:
    return available_groups()


def _load_group(group: str, data_dir: Optional[str]) -> list[EvalItem]:
    if group in _CURATED_GROUPS:
        return list(_CURATED_GROUPS[group])
    if group in _DISK_GROUPS:
        if not data_dir:
            raise ValueError(f"group {group!r} needs --data-dir (run fetch-eval-data)")
        if group == "seedtts_en":
            return load_seedtts_en(data_dir)
        return load_ellav_hard(os.path.join(data_dir, "ellav_hard.txt"))
    raise ValueError(f"unknown group: {group!r} (valid: {_valid_groups()})")


def iter_eval_items(
    active_groups: Iterable[str],
    max_per_group: Optional[int] = None,
    data_dir: Optional[str] = None,
) -> list[EvalItem]:
    """Flat, ordered work-list of EvalItems for the requested groups.

    ``max_per_group`` slices the first N of each group (smoke tests / tight VRAM).
    Order is stable across calls so shard partitioning is reproducible.
    """
    out: list[EvalItem] = []
    for group in active_groups:
        items = _load_group(group, data_dir)
        if max_per_group is not None:
            items = items[:max_per_group]
        out.extend(items)
    return out


def shard_cells(cells: list, num_shards: int, shard_id: int) -> list:
    """Return the ``shard_id`` slice of ``cells`` under round-robin partitioning.

    ``cell_index % num_shards == shard_id``. Over all ``shard_id`` the shards are
    disjoint and cover every cell, so parallel workers never drop or double-run one.
    """
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if not (0 <= shard_id < num_shards):
        raise ValueError(f"shard_id {shard_id} out of range [0, {num_shards})")
    return [cell for i, cell in enumerate(cells) if i % num_shards == shard_id]
