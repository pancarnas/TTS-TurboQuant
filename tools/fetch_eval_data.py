"""Download / vendor the evaluation data the benchmarks consume.

Populates a single ``--data-dir`` (default ``data/``) with BOTH standard sets so
``turboquant.eval_sentences`` can load them:

  - **seed-tts-eval test-en** — pulled from HuggingFace (``zhaochenyang20/seed-tts-eval``)
    into ``<data_dir>/en/*.lst`` + ``<data_dir>/prompt-wavs/`` (+ ``wavs/`` ground truth).
  - **ELLA-V hard sentences** — written to ``<data_dir>/ellav_hard.txt``. The ELLA-V
    *audio* prompts are not public, but only the *text* is needed (we supply our own
    voice), so this ships a clearly-labelled AR-stress reconstruction (tongue-twisters
    + repetitions). Replace the file with the canonical 100 if you obtain them.

Run: ``python tools/fetch_eval_data.py --data-dir data`` (or ``make fetch-eval-data``).
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

SEEDTTS_REPO = "zhaochenyang20/seed-tts-eval"
# The English set lives entirely under en/ (en/meta.lst + en/prompt-wavs + en/wavs);
# ** is recursive. Root-level fallbacks kept in case a mirror uses a flat layout.
_SEEDTTS_PATTERNS = ["en/**", "prompt-wavs/**", "wavs/**"]

# Reconstruction of the ELLA-V-style hard set: word repetitions + tongue-twisters,
# the patterns that make autoregressive TTS loop/skip (and that KV-compression
# amplifies). NOT the original 100 — do not compare WER head-to-head with ELLA-V.
_ELLAV_RECON = [
    "the great greek grape growers grow great greek grapes",
    "How many cookies could a good cook cook if a good cook could cook cookies? "
    "A good cook could cook as many cookies as a good cook who could cook cookies.",
    "Peter Piper picked a peck of pickled peppers; a peck of pickled peppers Peter Piper picked.",
    "She sells seashells by the seashore, and the shells she sells are surely seashells.",
    "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
    "Fuzzy Wuzzy was a bear, Fuzzy Wuzzy had no hair, so Fuzzy Wuzzy wasn't very fuzzy, was he?",
    "Red lorry, yellow lorry, red lorry, yellow lorry, red lorry, yellow lorry.",
    "I scream, you scream, we all scream, we all scream, we all scream for ice cream.",
    "The sixth sick sheikh's sixth sheep's sick, the sixth sick sheikh's sixth sheep's sick.",
    "Betty Botter bought some butter but she said the butter's bitter; if I put it in my batter it will make my batter bitter.",
    "A proper copper coffee pot, a proper copper coffee pot, a proper copper coffee pot.",
    "Six slippery snails slid slowly seaward, six slippery snails slid slowly seaward.",
    "Can you can a can as a canner can can a can? A canner can can a can the way a canner can.",
    "Three free throws, three free throws, three free throws, three free throws, three free throws.",
    "Truly rural, truly rural, truly rural, truly rural, truly rural, truly rural.",
    "Unique New York, you know you need unique New York, unique New York, you know you need unique New York.",
    "Which witch wished which wicked wish? The witch which wished the wicked wish wished it wickedly.",
    "Black bug's blood, black bug's blood, black bug's blood, black bug's blood, black bug's blood.",
    "The thirty-three thieves thought that they thrilled the throne throughout Thursday.",
    "Imagine an imaginary menagerie manager managing an imaginary menagerie.",
]


_SEEDTTS_LSTS = ("en/meta.lst", "en/non_para_reconstruct_meta.lst")


def _referenced_wavs(lst_path: str, limit: Optional[int]) -> list[str]:
    """Repo-relative wav paths referenced by the first ``limit`` rows of an .lst."""
    rels: list[str] = []
    rows = 0
    with open(lst_path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split("|")
            if len(parts) < 4:
                continue
            rels.append(parts[2])  # prompt wav
            if len(parts) >= 5 and parts[4].strip():
                rels.append(parts[4])  # ground-truth wav
            rows += 1
            if limit and rows >= limit:
                break
    return rels


def _resolve_real_paths(rels, repo_files) -> list[str]:
    """Map ``.lst`` wav refs to actual repo paths (they live under ``en/``, not root).

    The ``.lst`` lists ``prompt-wavs/x.wav`` relative to its own ``en/`` dir, so the
    real repo path is ``en/prompt-wavs/x.wav``. Match by exact path or path suffix so
    this works whatever prefix the dataset actually uses. Unmatched refs are dropped.
    """
    fileset = set(repo_files)
    out: list[str] = []
    for rel in rels:
        if rel in fileset:
            out.append(rel)
            continue
        match = next((f for f in repo_files if f.endswith("/" + rel)), None)
        if match:
            out.append(match)
    return out


def _parallel_download(repo: str, rels, data_dir: str, token, workers: int) -> None:
    """Download repo-relative files concurrently (token raises the anon rate limit)."""
    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import hf_hub_download

    def _one(rel: str) -> None:
        hf_hub_download(
            repo_id=repo,
            repo_type="dataset",
            filename=rel,
            local_dir=data_dir,
            token=token,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(_one, rels))


def fetch_seedtts(data_dir: str, limit: Optional[int] = None, workers: int = 4) -> int:
    """Download seed-tts-eval test-en into ``data_dir``; return en sample count.

    Disables Xet (its anonymous ``xet-read-token`` endpoint rate-limits hard — the
    common 429), so it uses plain HTTPS. With ``limit`` set, fetches only the wav
    clips the first N rows reference (≈100-200 files, not all ~2,170) — far faster
    and rate-limit-safe. ``workers`` controls download concurrency. An HF token
    (``HF_TOKEN`` / cached login) raises the anonymous limit; set one for 429s.
    """
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    try:
        from huggingface_hub import HfApi, hf_hub_download, snapshot_download
        from huggingface_hub.errors import HfHubHTTPError
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise SystemExit(
            "huggingface_hub not installed (uv add huggingface_hub)"
        ) from exc
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    try:
        # The .lst index files are tiny — always grab them first.
        for lst in _SEEDTTS_LSTS:
            hf_hub_download(
                repo_id=SEEDTTS_REPO,
                repo_type="dataset",
                filename=lst,
                local_dir=data_dir,
                token=token,
            )
        if limit:
            rels: set[str] = set()
            for lst in _SEEDTTS_LSTS:
                p = os.path.join(data_dir, lst)
                if os.path.exists(p):
                    rels.update(_referenced_wavs(p, limit))
            # The .lst paths are relative to en/; resolve to the real repo paths.
            repo_files = HfApi().list_repo_files(
                repo_id=SEEDTTS_REPO, repo_type="dataset", token=token
            )
            real = _resolve_real_paths(sorted(rels), repo_files)
            print(f"  fetching {len(real)} wavs for the first {limit} samples ...")
            _parallel_download(SEEDTTS_REPO, real, data_dir, token, workers)
        else:
            snapshot_download(
                repo_id=SEEDTTS_REPO,
                repo_type="dataset",
                local_dir=data_dir,
                allow_patterns=_SEEDTTS_PATTERNS,
                token=token,
                max_workers=workers,
            )
    except HfHubHTTPError as exc:
        if "429" in str(exc):
            raise SystemExit(
                "HF rate-limited this IP (429). Either pass --limit N (fetch only "
                "the wavs you need) or authenticate to raise the limit:\n"
                "  huggingface-cli login   # or: export HF_TOKEN=hf_...\n"
                "Xet is already disabled via HF_HUB_DISABLE_XET=1."
            ) from exc
        raise
    meta = os.path.join(data_dir, "en", "meta.lst")
    if not os.path.exists(meta):
        return 0
    with open(meta, encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def write_ellav_hard(data_dir: str, overwrite: bool = False) -> int:
    """Write the vendored ELLA-V-style hard set; return sentence count."""
    path = os.path.join(data_dir, "ellav_hard.txt")
    if os.path.exists(path) and not overwrite:
        with open(path, encoding="utf-8") as fh:
            return sum(1 for line in fh if line.strip() and not line.startswith("#"))
    header = (
        "# ELLA-V-style hard sentences (RECONSTRUCTION, not the original 100).\n"
        "# AR-stress patterns: word repetitions + tongue-twisters. Provenance: vendored\n"
        "# by tools/fetch_eval_data.py. Do NOT compare WER head-to-head with ELLA-V.\n"
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(header)
        fh.write("\n".join(_ELLAV_RECON) + "\n")
    return len(_ELLAV_RECON)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", default="data", help="Target dir (default: data)."
    )
    parser.add_argument(
        "--skip-seedtts", action="store_true", help="Only (re)write the ELLA-V text."
    )
    parser.add_argument(
        "--overwrite-ellav",
        action="store_true",
        help="Overwrite an existing ellav_hard.txt.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Fetch only the wav clips the first N seed-tts rows reference "
        "(≈100-200 files vs all ~2,170). Match this to the --max-per-group you "
        "will run. Default: download the whole test-en set.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel download workers (default 4). Raise with an HF token set.",
    )
    args = parser.parse_args()
    os.makedirs(args.data_dir, exist_ok=True)

    print(f"== Eval-data fetch into {args.data_dir} ==")
    seed_n = 0
    if not args.skip_seedtts:
        seed_n = fetch_seedtts(args.data_dir, limit=args.limit, workers=args.workers)
        print(
            f"seed-tts-eval test-en: {seed_n} samples indexed  (source: HF {SEEDTTS_REPO})"
        )
    ellav_n = write_ellav_hard(args.data_dir, overwrite=args.overwrite_ellav)
    print(f"ellav_hard: {ellav_n} sentences  (source: vendored reconstruction)")
    print(
        "Done. Pass --data-dir to the benchmarks to use seedtts_en / ellav_hard groups."
    )


if __name__ == "__main__":
    main()
