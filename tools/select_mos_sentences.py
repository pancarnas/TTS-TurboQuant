"""Pick a speaker-diverse librispeech_pc sentence set for the MOS study.

The eval `.lst` groups utterances by speaker, so the first 100 idxs cover only ~8
of the 39 speakers. This tool selects up to ``--per-speaker`` sentences from EACH
speaker (in a 4-9 s length window) so the listening study spans all voices, and
writes their idxs (one per line) for ``benchmark_vallex_real.py --idx-file`` /
``vallex_ppl_divergence.py --idx-file``.

idx = position in the eval list. We parse the `.lst` directly, mirroring
``parse_librispeech_pc_lst`` (TAB, skip <6-field lines), so the enumeration matches
``iter_eval_items(["librispeech_pc"], None, data_dir)`` exactly; when torch is
importable the counts are cross-checked. Torch-free otherwise (plain text parse) so
it can be sanity-run anywhere the `.lst` is present.

  python tools/select_mos_sentences.py --data-dir data --per-speaker 5 \
      --min-dur 4 --max-dur 9 --out mos_idx_list.txt
"""

from __future__ import annotations

import argparse
import os

GROUP = "librispeech_pc"
DEFAULT_LST = "librispeech_pc_test_clean_cross_sentence.lst"


def _parse_lst(lst_path: str) -> list[tuple[int, str, float]]:
    """(idx, speaker, gen_dur) per accepted line, mirroring parse_librispeech_pc_lst.

    Row: ref_utt \\t ref_dur \\t ref_text \\t gen_utt \\t gen_dur \\t gen_text.
    Speaker = the target (gen) utterance's speaker; dur = the target duration.
    Lines with <6 TAB fields are skipped, so idx stays aligned with the loader.
    """
    out: list[tuple[int, str, float]] = []
    idx = 0
    with open(lst_path, encoding="utf-8") as fh:
        for raw in fh:
            parts = raw.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            gen_utt, gen_dur = parts[3].strip(), parts[4].strip()
            speaker = gen_utt.split("-")[0]
            try:
                dur = float(gen_dur)
            except ValueError:
                dur = float("nan")
            out.append((idx, speaker, dur))
            idx += 1
    return out


def _crosscheck(n_lines: int, data_dir: str) -> None:
    """Assert the loader yields the same item count (skipped if torch unavailable)."""
    try:
        from turboquant.eval_sentences import iter_eval_items
    except Exception as exc:  # torch/lib not importable here — fine, skip.
        print(f"(skipped iter_eval_items cross-check: {type(exc).__name__})")
        return
    n_items = len(iter_eval_items([GROUP], None, data_dir))
    if n_items != n_lines:
        raise SystemExit(
            f"ALIGNMENT MISMATCH: parsed {n_lines} lines but iter_eval_items has "
            f"{n_items}; idx would not line up with generation/scoring. Aborting."
        )
    print(f"alignment OK: {n_items} items match parsed lines")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--lst-name", default=DEFAULT_LST)
    ap.add_argument("--per-speaker", type=int, default=5)
    ap.add_argument("--min-dur", type=float, default=4.0)
    ap.add_argument("--max-dur", type=float, default=9.0)
    ap.add_argument("--out", default="mos_idx_list.txt")
    args = ap.parse_args()

    lst_path = os.path.join(args.data_dir, GROUP, args.lst_name)
    rows = _parse_lst(lst_path)
    _crosscheck(len(rows), args.data_dir)

    # First --per-speaker in-range sentences per speaker, in list order (deterministic).
    picked: dict[str, list[int]] = {}
    for idx, spk, dur in rows:
        if not (args.min_dur <= dur <= args.max_dur):
            continue
        lst = picked.setdefault(spk, [])
        if len(lst) < args.per_speaker:
            lst.append(idx)

    chosen = sorted(i for lst in picked.values() for i in lst)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(str(i) for i in chosen) + "\n")

    speakers = sorted(picked)
    short = [s for s in speakers if len(picked[s]) < args.per_speaker]
    print(f"wrote {len(chosen)} idxs across {len(speakers)} speakers -> {args.out}")
    print(f"  target {args.per_speaker}/speaker; "
          f"{len(speakers) - len(short)} speakers full, {len(short)} short")
    if short:
        print("  short speakers (speaker=count): "
              + ", ".join(f"{s}={len(picked[s])}" for s in short))
    # total speakers present in the corpus (for context)
    all_spk = {spk for _, spk, _ in rows}
    missing = sorted(all_spk - set(speakers))
    if missing:
        print(f"  {len(missing)} speakers had NO 4-9 s sentence: {missing}")


if __name__ == "__main__":
    main()
