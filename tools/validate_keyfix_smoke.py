"""Validate a 03_smoke_keyfix.sh run (no regeneration): pipeline correctness only.

Reads results/smoke_keyfix_{plain,g32,pre}.csv and the smoke wav dir and prints
a PASS/FAIL line per check plus one verdict. It deliberately asserts only what
MUST hold if the plumbing is right — it never asserts the scientific outcome
(whether group scaling / pre-RoPE help), which the full run is for.

  python tools/validate_keyfix_smoke.py --results results \
      --wavdir models/Qwen3-TTS/benchmarks/outputs/smoke_keyfix --n 2
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys

CELLS = {"plain": (0, 0), "g32": (32, 0), "pre": (0, 1)}


def _sha(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha1(fh.read()).hexdigest()


def _wav(wavdir, idx, cfg):
    """Smoke wav path for sentence idx and config label 'fp16' | (kb, vb, cell)."""
    if cfg == "fp16":
        name = f"qwen_librispeech_pc_{idx}_sampling_s0_t0.9_fp16.wav"
    else:
        kb, vb, cell = cfg
        seg = f"K{kb}_V{vb}_rw=0_pl=0"
        if cell == "g32":
            seg += "_g=32"
        if cell == "pre":
            seg += "_pre"
        name = f"qwen_librispeech_pc_{idx}_sampling_s0_t0.9_{seg}.wav"
    return os.path.join(wavdir, name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results")
    ap.add_argument("--wavdir", default="models/Qwen3-TTS/benchmarks/outputs/smoke_keyfix")
    ap.add_argument("--n", type=int, default=2, help="sentences per cell")
    ap.add_argument("--logs", default="logs",
                    help="dir with smoke_keyfix_<cell>.log (measurement-error scan; skipped if absent)")
    args = ap.parse_args()

    import pandas as pd

    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    frames = {}
    for cell, (kg, pre) in CELLS.items():
        p = os.path.join(args.results, f"smoke_keyfix_{cell}.csv")
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            check(f"{cell}: divergence CSV exists ({p})", False)
            continue
        df = pd.read_csv(p)
        frames[cell] = df
        check(f"{cell}: has rows ({len(df)})", len(df) > 0)
        check(f"{cell}: key_group/pre_rope columns present", {"key_group", "pre_rope"} <= set(df.columns))
        if {"key_group", "pre_rope"} <= set(df.columns):
            check(f"{cell}: key_group=={kg} and pre_rope=={pre} on every row",
                  (df.key_group == kg).all() and (df.pre_rope == pre).all())
        check(f"{cell}: both K8V8 and K2V2 recorded",
              {8, 2} <= set(df.key_bits.unique()))
        check(f"{cell}: {args.n} sentences", df.idx.nunique() == args.n)

    # A measurement error inside record() is caught and counted, not raised, so
    # a cell can look fine while most of its rows are missing (first smoke run:
    # the pre cell kept 22k of 115k rows). Every cell runs the same sentences,
    # configs and steps, so the row counts must agree exactly.
    if len(frames) == 3:
        n_plain = len(frames["plain"])
        for cell in ("g32", "pre"):
            check(f"{cell}: same row count as plain ({len(frames[cell])} vs {n_plain})",
                  len(frames[cell]) == n_plain)
    for cell in CELLS:
        lp = os.path.join(args.logs, f"smoke_keyfix_{cell}.log")
        if not os.path.exists(lp):
            continue
        with open(lp, errors="replace") as fh:
            m = re.search(r"WARNING: (\d+) measurement errors", fh.read())
        n_err = int(m.group(1)) if m else 0
        check(f"{cell}: no measurement errors in {lp} ({n_err})", n_err == 0)

    if len(frames) == 3:
        for cell, df in frames.items():
            g = df.groupby("key_bits")[["cos_k", "cos_v", "attn_js", "relmse_k"]].mean()
            print(f"\n  {cell} — mean by key_bits:\n" + g.round(5).to_string())
        g8 = frames["pre"][frames["pre"].key_bits == 8]
        # The decisive E2 plumbing check: with 8-bit keys the only error that
        # could remain is a wrong RoPE undo/redo on the real mRoPE tables.
        check("pre: K8V8 pre-RoPE cos_k > 0.999 (RoPE undone exactly)", g8.cos_k.mean() > 0.999)
        check("pre: K8V8 pre-RoPE attn_js < 1e-3", g8.attn_js.mean() < 1e-3)
        p8 = frames["plain"][frames["plain"].key_bits == 8]
        check("plain: K8V8 cos_k > 0.999 (sanity)", p8.cos_k.mean() > 0.999)
        c2 = {c: frames[c][frames[c].key_bits == 2].cos_k.mean() for c in CELLS}
        check("all cells: K2V2 cos_k in [0.5, 0.999] (lossy, not broken)",
              all(0.5 <= v <= 0.999 for v in c2.values()))
        # informational — this is the hypothesis the full run tests
        print(f"  [info] K2V2 cos_k   plain={c2['plain']:.4f}  g32={c2['g32']:.4f}  pre={c2['pre']:.4f}")
        j2 = {c: frames[c][frames[c].key_bits == 2].attn_js.mean() for c in CELLS}
        print(f"  [info] K2V2 attn_js plain={j2['plain']:.5f}  g32={j2['g32']:.5f}  pre={j2['pre']:.5f}")

    # --- wavs ---
    idxs = sorted(frames["plain"].idx.unique()) if "plain" in frames else list(range(args.n))
    for idx in idxs:
        fp = _wav(args.wavdir, idx, "fp16")
        check(f"idx {idx}: fp16 wav exists", os.path.exists(fp))
        hashes = {}
        for cell in CELLS:
            p = _wav(args.wavdir, idx, (2, 2, cell))
            if os.path.exists(p):
                hashes[cell] = _sha(p)
            check(f"idx {idx}: K2V2 wav exists for cell {cell}", os.path.exists(p))
            p8 = _wav(args.wavdir, idx, (8, 8, cell))
            check(f"idx {idx}: K8V8 wav exists for cell {cell}", os.path.exists(p8))
        if len(hashes) == 3:
            # the flags must reach generation: a flag that changed nothing is a no-op bug
            check(f"idx {idx}: K2V2 wav differs plain vs g32", hashes["plain"] != hashes["g32"])
            check(f"idx {idx}: K2V2 wav differs plain vs pre", hashes["plain"] != hashes["pre"])
    # fp16 is written by every cell to the SAME path (no flag segment); it was
    # overwritten three times with the same seed — its CSV-free check is that
    # the K2V2 wavs above differ while fp16 exists. (Byte-identity across cells
    # cannot be checked from one file; the no-flag-segment naming guarantees
    # the scorer pairs every cell against one baseline.)

    print("\n===== KEYFIX SMOKE " + ("PASS — safe to launch 14_qwen_keyfix.sh"
                                      if ok else "FAIL — inspect above before launching") + " =====")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
