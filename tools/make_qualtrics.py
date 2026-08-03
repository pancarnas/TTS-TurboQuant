"""Generate a Qualtrics Advanced-Format survey from the MOS/SMOS manifest.

Reads mos_study_clone/manifest.csv (clip_id, block, role, system, sentence_idx,
ref_clip_id, ...) and a hosting base URL, and writes a .txt that Qualtrics
imports (Survey -> Tools -> Import/Export -> Import Survey, "Advanced Format").
Clips are expected at <base-url>/<clip_id>.wav.

One block per group. Each stimulus is its own page: a 5-point naturalness (MOS)
question first, then - for synthesized systems - a 5-point speaker-similarity
(SMOS) question on the same page that reveals the sentence's reference clip and
asks about "the clip above" (naturalness-first ordering limits reference
anchoring). Pages are separated by a page break. Question IDs are ``nat_<clip>`` /
``sim_<clip>`` so the export joins straight back to the manifest.

  python tools/make_qualtrics.py \
      --manifest mos_study_clone/manifest.csv \
      --base-url https://YOUR_HOST/mos_clips \
      --out mos_study_clone/qualtrics_import.txt

Stimulus order is already shuffled + anonymized at build time, so DO NOT enable
per-question randomisation (it would split each naturalness/similarity pair across
pages). In Survey Flow add a Randomizer that presents 1 of the N blocks per
respondent (evenly, with "Evenly Present Elements"). Attention checks are the
natural clips (should score high) and the K2V2@0 clips (should score low) -
identify them via the manifest.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

NAT_CHOICES = ["1 - Bad", "2 - Poor", "3 - Fair", "4 - Good", "5 - Excellent"]
SIM_CHOICES = ["1 - Not at all similar", "2 - Slightly similar",
               "3 - Moderately similar", "4 - Very similar",
               "5 - Identical speaker"]


def _audio(url: str) -> str:
    return (f'<audio controls preload="none" style="width:320px">'
            f'<source src="{url}" type="audio/wav"></audio>')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="mos_study_clone/manifest.csv")
    ap.add_argument("--base-url", required=True,
                    help="base URL for the clips; each is <base-url>/<clip_id>.wav")
    ap.add_argument("--out", default="mos_study_clone/qualtrics_import.txt")
    ap.add_argument("--similarity-configs", default="",
                    help="comma list of systems that get the similarity (SMOS) "
                         "question, e.g. 'fp16,K4V2@0,K2V2@0'; empty = all "
                         "synthesized systems")
    ap.add_argument("--natural-similarity", action="store_true",
                    help="also ask similarity for the natural clips (high-anchor "
                         "attention check); default excludes them")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    base = args.base_url.rstrip("/")
    stim = df[df["role"] == "stimulus"].copy()
    sim_set = {c.strip() for c in args.similarity_configs.split(",") if c.strip()}

    lines: list[str] = ["[[AdvancedFormat]]", ""]
    n_nat = n_sim = 0
    for block in sorted(stim["block"].unique()):
        sub = stim[stim["block"] == block]
        letter = chr(65 + int(block))

        # one merged block per group; each stimulus is its own page (naturalness,
        # then similarity for SMOS systems), separated by a page break.
        lines += [f"[[Block:Block {letter}]]", ""]
        for _, r in sub.iterrows():
            # --- naturalness (MOS): every stimulus, rated first ---
            surl = f"{base}/{r['clip_id']}.wav"
            lines += [
                "[[Question:MC:SingleAnswer]]",
                f"[[ID:nat_{r['clip_id']}]]",
                f"{_audio(surl)}<br><br>Rate the overall quality and naturalness "
                "of this speech.",
                "[[Choices]]", *NAT_CHOICES, "",
            ]
            n_nat += 1

            # --- similarity (SMOS): same page, reference revealed after ---
            ref = r.get("ref_clip_id")
            emit_sim = isinstance(ref, str) and bool(ref)
            if r["system"] == "natural" and not args.natural_similarity:
                emit_sim = False
            if sim_set and r["system"] not in sim_set:
                emit_sim = False
            if emit_sim:
                rurl = f"{base}/{ref}.wav"
                lines += [
                    "[[Question:MC:SingleAnswer]]",
                    f"[[ID:sim_{r['clip_id']}]]",
                    f"Reference voice:<br>{_audio(rurl)}<br><br>"
                    "How similar is the speaker in the clip <b>above</b> to this "
                    "<b>reference voice</b>?",
                    "[[Choices]]", *SIM_CHOICES, "",
                ]
                n_sim += 1

            lines += ["[[PageBreak]]", ""]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    n_blocks = stim["block"].nunique()
    print(f"wrote {args.out}")
    print(f"  {n_blocks} blocks, {n_nat} naturalness + {n_sim} similarity "
          f"= {n_nat + n_sim} questions")
    print(f"  per block ~= {(n_nat + n_sim) // max(n_blocks, 1)} questions "
          "(one respondent does one block)")


if __name__ == "__main__":
    main()
