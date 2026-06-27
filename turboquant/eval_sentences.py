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
from dataclasses import dataclass, replace
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
    lines: Iterable[str],
    audio_root: str,
    group: str = "seedtts_en",
    delimiter: str = "|",
) -> list[EvalItem]:
    """Parse a seed-tts-style ``.lst`` into EvalItems.

    Each row is ``id <d> prompt_text <d> prompt_wav <d> target_text`` (4 fields)
    and a cross-speaker set adds a 5th ``target_wav`` (ground truth), where ``<d>``
    is ``delimiter`` (``|`` for seed-tts, ``\\t`` for the F5-TTS LibriSpeech-PC
    list). Relative wav paths resolve against ``audio_root``; blank/short
    (<4 field) lines are skipped.
    """
    items: list[EvalItem] = []
    for raw in lines:
        line = raw.rstrip("\n")
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(delimiter)]
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


def _detect_delimiter(lst_path: str) -> str:
    """Sniff ``|`` vs tab from the first non-comment line of a meta ``.lst``."""
    with open(lst_path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if line and not line.startswith("#"):
                return "|" if "|" in line else "\t"
    return "|"


def _libri_flac_path(libri_root: str, utt_id: str) -> str:
    """LibriSpeech utterance id ``SPK-CHAPTER-UTT`` → ``<root>/SPK/CHAPTER/ID.flac``."""
    spk, chapter = utt_id.split("-")[:2]
    return os.path.join(libri_root, spk, chapter, f"{utt_id}.flac")


def parse_librispeech_pc_lst(
    lines: Iterable[str], libri_root: str, group: str = "librispeech_pc"
) -> list[EvalItem]:
    """Parse the F5-TTS LibriSpeech-PC cross-sentence ``.lst`` (TAB, 6 fields).

    Each row is ``ref_utt \\t ref_dur \\t ref_text \\t gen_utt \\t gen_dur \\t gen_text``:
    ``ref_*`` is the **prompt** (reference voice) and ``gen_*`` is the **target**
    (what to synthesize). Utterance ids resolve to flac under ``libri_root``
    (LibriSpeech test-clean), so each item gets its own real prompt (ref) and the
    real target recording (ground truth). Short/malformed lines are skipped.
    """
    items: list[EvalItem] = []
    for raw in lines:
        parts = raw.rstrip("\n").split("\t")
        if len(parts) < 6:
            continue
        ref_utt, _ref_dur, ref_text, gen_utt, _gen_dur, gen_text = parts[:6]
        items.append(
            EvalItem(
                text=gen_text.strip(),
                group=group,
                ref_audio=_libri_flac_path(libri_root, ref_utt.strip()),
                ref_text=ref_text.strip(),
                ground_truth_audio=_libri_flac_path(libri_root, gen_utt.strip()),
            )
        )
    return items


def load_librispeech_pc(
    data_dir: str,
    limit: Optional[int] = None,
    lst_name: str = "librispeech_pc_test_clean_cross_sentence.lst",
    libri_subdir: str = os.path.join("LibriSpeech", "test-clean"),
) -> list[EvalItem]:
    """Load the F5-TTS LibriSpeech-PC test-clean cross-sentence set.

    Standard zero-shot-TTS list (1,127 pairs / 39 speakers, 4–10 s). The ``.lst``
    lives at ``<data_dir>/librispeech_pc/<lst_name>``; flac audio resolves under
    ``<data_dir>/librispeech_pc/<libri_subdir>`` (the unpacked OpenSLR test-clean).
    """
    root = os.path.join(data_dir, "librispeech_pc")
    lst_path = os.path.join(root, lst_name)
    libri_root = os.path.join(root, libri_subdir)
    with open(lst_path, encoding="utf-8") as fh:
        items = parse_librispeech_pc_lst(fh, libri_root, group="librispeech_pc")
    return items[:limit] if limit else items


def load_libritts_long(data_dir: str, limit: Optional[int] = None) -> list[EvalItem]:
    """Load the long-context set built from LibriTTS-R by concatenation.

    Thin parser over the manifest ``<data_dir>/libritts_long/manifest.lst``
    written by ``tools/build_libritts_long.py`` (which does the audio stitching);
    the concatenation logic itself is the pure, tested ``plan_long_concatenations``.
    Each row carries a same-speaker prompt (ref) + the concatenated real recording
    (ground truth).
    """
    root = os.path.join(data_dir, "libritts_long")
    lst_path = os.path.join(root, "manifest.lst")
    with open(lst_path, encoding="utf-8") as fh:
        items = parse_seedtts_lst(
            fh,
            audio_root=root,
            group="libritts_long",
            delimiter=_detect_delimiter(lst_path),
        )
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


# ---------------------------------------------------------------------------
# Length heuristic + difficulty classifier (Experiment 3: 3×3 length×difficulty)
# ---------------------------------------------------------------------------
#
# Empirical fit on Exp-1 baseline generations (Qwen3-TTS-12Hz, R²≈0.995):
# generated talker tokens ≈ 6.4 × words ≈ 1.05 × chars. At 12 Hz that is
# ≈0.53 s/word. Used to (a) bin sentences into short/medium/long and (b) target
# curated lengths so compression bites across the whole range at rw=0.

TOKENS_PER_WORD = 7.0
SHORT_MAX_TOKENS = 96  # ≈15 words — boundary sits in the gap above the short band
MEDIUM_MAX_TOKENS = 384  # ≈60 words — boundary sits in the gap below the long band

# Cell target bands (predicted tokens) the curated/filtered sets aim for.
LENGTH_TARGET_TOKENS = {
    "short": (32, 64),
    "medium": (128, 256),
    "long": (512, 1024),
}

_DIFFICULTIES = ("easy", "medium", "hard")
_SUBORDINATE = re.compile(
    r"\b(which|because|although|though|while|whereas|however|despite|"
    r"unless|whenever|therefore|moreover)\b",
    re.IGNORECASE,
)
_ABBREV = re.compile(r"\b[A-Z]{2,}\b")
_PUNCT = ".,!?;:'\"()"
_CODE_SYMBOLS = "%$#&/@"


def _code_token_count(text: str) -> int:
    """Count distinct code-like TOKENS + acronyms, not characters.

    A single multi-digit number ("1859", a ZIP, a price) is ONE token, so it does
    not by itself read as digit-density. Tongue-twister/digit 'hard' density means
    several separate numeric/code/URL/acronym tokens — this counts those.
    """
    n = 0
    for tok in text.split():
        if (
            any(c.isdigit() for c in tok)
            or any(s in tok for s in _CODE_SYMBOLS)
            or "http" in tok
            or "www." in tok
        ):
            n += 1
    return n + len(_ABBREV.findall(text))


def predict_tokens(text: str) -> int:
    """Predicted Qwen3-TTS-12Hz talker tokens for ``text`` (≈6.4 × words)."""
    return round(TOKENS_PER_WORD * len(text.split()))


def length_category(text: str) -> str:
    """Bin ``text`` into ``short`` / ``medium`` / ``long`` by predicted tokens."""
    tokens = predict_tokens(text)
    if tokens <= SHORT_MAX_TOKENS:
        return "short"
    if tokens <= MEDIUM_MAX_TOKENS:
        return "medium"
    return "long"


LONG_BUCKETS = (256, 512, 1024, 2048)


def plan_long_concatenations(
    utterances: list[dict], buckets: tuple[int, ...] = LONG_BUCKETS
) -> list[dict]:
    """Plan long-context passages by concatenating consecutive same-chapter utts.

    Pure (no I/O) so it is unit-testable; the audio stitching that consumes this
    lives in ``tools/build_libritts_long.py``. ``utterances`` are dicts with keys
    ``speaker, chapter, idx, text, wav``. For each target ``bucket`` (predicted
    talker tokens), walk each chapter in ``idx`` order and emit **non-overlapping**
    passages whose cumulative ``predict_tokens`` first reaches the bucket. Each
    passage gets a same-speaker **prompt** (``ref_wav``/``ref_text``) drawn from an
    utterance NOT in the passage (a different chapter when possible), so the clone
    prompt never leaks the target. Returns records with ``target_text``,
    ``member_wavs``, ``ref_wav``, ``ref_text``, ``speaker``, ``bucket``,
    ``actual_tokens``.
    """
    by_chapter: dict[tuple, list[dict]] = {}
    by_speaker: dict[str, list[dict]] = {}
    for u in utterances:
        by_chapter.setdefault((u["speaker"], u["chapter"]), []).append(u)
        by_speaker.setdefault(u["speaker"], []).append(u)
    for group in by_chapter.values():
        group.sort(key=lambda u: u["idx"])

    def _pick_ref(speaker: str, member_wavs: set) -> Optional[dict]:
        # Prefer a same-speaker utterance from a different chapter; else any
        # same-speaker utterance not used in this passage.
        pool = [u for u in by_speaker.get(speaker, []) if u["wav"] not in member_wavs]
        return pool[0] if pool else None

    out: list[dict] = []
    for bucket in buckets:
        for (speaker, _), utts in by_chapter.items():
            acc: list[dict] = []
            acc_tokens = 0
            for u in utts:
                acc.append(u)
                acc_tokens += predict_tokens(u["text"])
                if acc_tokens >= bucket:
                    member_wavs = {m["wav"] for m in acc}
                    ref = _pick_ref(speaker, member_wavs)
                    out.append(
                        {
                            "speaker": speaker,
                            "bucket": bucket,
                            "actual_tokens": acc_tokens,
                            "target_text": " ".join(m["text"] for m in acc),
                            "member_wavs": [m["wav"] for m in acc],
                            "ref_wav": ref["wav"] if ref else None,
                            "ref_text": ref["text"] if ref else None,
                        }
                    )
                    acc, acc_tokens = [], 0
    return out


def _has_proper_noun(words: list[str]) -> bool:
    """True if a capitalized word appears NOT at a sentence start.

    Tracks sentence boundaries so the capital after a period (a new sentence) is
    not mistaken for a proper noun — essential for multi-sentence easy passages,
    which would otherwise all read as 'medium'.
    """
    sentence_start = True
    for word in words:
        stripped = word.strip(_PUNCT)
        if not stripped:
            continue
        if stripped[0].isupper() and not sentence_start and stripped != "I":
            return True
        sentence_start = word.endswith((".", "!", "?"))
    return False


def text_difficulty(text: str) -> str:
    """Classify text-intrinsic difficulty from surface features (no model needed).

    ``hard`` = AR-stress: adjacent word repetitions, strong content-word
    alliteration, or digit/URL/abbreviation density. ``medium`` = proper nouns,
    numbers, or subordinate clauses. ``easy`` = plain high-frequency declaratives.
    Used to split the natural seedtts pool and to *validate* the easy<medium<hard
    baseline-CER ordering of the curated cells.
    """
    words = text.split()
    lowered = [w.strip(_PUNCT).lower() for w in words]

    # hard: immediate repetition (e.g. "the the", "said said")
    if any(a and a == b for a, b in zip(lowered, lowered[1:])):
        return "hard"
    # hard: alliteration = a RUN of >=4 consecutive words (in original order)
    # sharing a first letter — a tongue-twister. Tiny function words (<=2 chars:
    # a, of, to, in, is, it…) are transparent (neither extend nor break the run),
    # but any other differently-lettered word resets it. This matches real
    # twisters ("Peter picked a peck of pickled peppers") without firing on long
    # prose, where coincidental same-letter words are separated by ordinary words.
    best = run = 0
    run_letter = ""
    for w in lowered:
        if len(w) <= 2:
            continue  # transparent function word
        if w[0] == run_letter:
            run += 1
        else:
            run, run_letter = 1, w[0]
        best = max(best, run)
    if best >= 4:
        return "hard"
    code_tokens = _code_token_count(text)
    if code_tokens >= 3:
        return "hard"

    # medium: any number/symbol, a real proper noun, or a subordinate clause
    if code_tokens >= 1 or _has_proper_noun(words) or _SUBORDINATE.search(text):
        return "medium"
    return "easy"


# The 3×3 experiment grid, encoded as flat group strings so the benchmark
# (`group` is a free axis) and analysis bucket by cell with no extra code.
_GRID_CELLS = tuple(
    f"{length}_{diff}"
    for length in ("short", "medium", "long")
    for diff in _DIFFICULTIES
)
# Natural cells come from seedtts (own ref audio); the rest are curated text
# cloned from a default reference.
_SEEDTTS_CELLS = ("short_easy", "short_medium")
_CURATED_CELLS = tuple(c for c in _GRID_CELLS if c not in _SEEDTTS_CELLS)


def load_curated_cell(data_dir: str, cell: str) -> list[EvalItem]:
    """Load curated sentences for a grid cell from ``<data_dir>/curated/<cell>.txt``.

    One sentence per line; ``#`` lines are comments. ``ref_audio`` is left None —
    the benchmark's clone path supplies the shared ``--default-ref-audio`` clip.
    """
    path = os.path.join(data_dir, "curated", f"{cell}.txt")
    items: list[EvalItem] = []
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            items.append(EvalItem(text=line, group=cell))
    return items


def load_seedtts_cell(data_dir: str, cell: str) -> list[EvalItem]:
    """Natural seedtts items matching a grid cell's length band + difficulty.

    Re-tags the matched items to ``cell`` (keeping their own ref audio/text) so
    trial rows record the cell. ``cell`` is e.g. ``short_easy``.
    """
    length, difficulty = cell.split("_", 1)
    pool = load_seedtts_en(data_dir)
    return [
        replace(item, group=cell)
        for item in pool
        if length_category(item.text) == length
        and text_difficulty(item.text) == difficulty
    ]


# Standard-corpus sets for the paper-foundation run (length sweep).
_STANDARD_GROUPS = ("librispeech_pc", "libritts_long")


def _assign_rotating_refs(
    items: list[EvalItem], data_dir: str, pool_size: int = 8
) -> list[EvalItem]:
    """Give text-only items (ELLA-V) references by rotating a clean prompt pool.

    Draws the first ``pool_size`` LibriSpeech-PC prompts as the reference pool and
    assigns ``pool[idx % n]`` so the AR-stress sentences clone from real, diverse
    speakers. If LibriSpeech-PC isn't fetched, items keep ``ref_audio=None`` and
    fall back to the run's ``--default-ref-audio``.
    """
    try:
        pool = load_librispeech_pc(data_dir, limit=pool_size)
    except (FileNotFoundError, OSError):
        return items
    if not pool:
        return items
    return [
        replace(
            item,
            ref_audio=pool[i % len(pool)].ref_audio,
            ref_text=pool[i % len(pool)].ref_text,
        )
        for i, item in enumerate(items)
    ]


def available_groups() -> list[str]:
    """All valid group names (standard sets + curated literals + grid cells)."""
    return (
        list(_DISK_GROUPS)
        + list(_STANDARD_GROUPS)
        + list(_CURATED_GROUPS)
        + list(_GRID_CELLS)
    )


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
        ella = load_ellav_hard(os.path.join(data_dir, "ellav_hard.txt"))
        return _assign_rotating_refs(ella, data_dir)
    if group in _STANDARD_GROUPS:
        if not data_dir:
            raise ValueError(f"group {group!r} needs --data-dir (run fetch-eval-data)")
        if group == "librispeech_pc":
            return load_librispeech_pc(data_dir)
        return load_libritts_long(data_dir)
    if group in _GRID_CELLS:
        if not data_dir:
            raise ValueError(f"group {group!r} needs --data-dir (run fetch-eval-data)")
        if group in _SEEDTTS_CELLS:
            return load_seedtts_cell(data_dir, group)
        return load_curated_cell(data_dir, group)
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
