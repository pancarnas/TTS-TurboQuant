"""Validate an existing smoke run (no regeneration): metric correctness + PASS/FAIL.

Checks the divergence CSV and score CSV from a smoke_full run assert internally
correct metrics (cos_k tracks key bits, cos_v tracks value bits & is
key-independent, attn_js ordering, seeds saved, English-preset WER sane).

  python tools/validate_smoke.py \
      --div results/smoke_full_div.csv \
      --scores results/smoke_full_scores.csv \
      --wavdir models/VALL-E-X/benchmarks/outputs/smoke_full --n 4
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--div", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--wavdir", required=True)
    ap.add_argument("--n", type=int, default=4, help="sentences per group")
    args = ap.parse_args()

    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    wavs = glob.glob(os.path.join(args.wavdir, "*.wav"))
    check(f"wav count == {args.n * 10}", len(wavs) == args.n * 10)
    seeds = {
        int(m.group(1))
        for f in wavs
        if (m := re.search(r"_s(\d+)_", os.path.basename(f)))
    }
    check("both seeds saved (0 and 1)", {0, 1} <= seeds)

    d = pd.read_csv(args.div)
    d["cfg"] = "K" + d.key_bits.astype(str) + "V" + d.value_bits.astype(str)
    g = d.groupby("cfg")[["cos_k", "cos_v", "attn_js", "relmse_k", "relmse_v"]].mean()
    print(g.round(4).to_string())
    check("cos_k: K4V4 > K2V2  (key bits drive cos_k)",
          g.loc["K4V4", "cos_k"] > g.loc["K2V2", "cos_k"] + 0.01)
    check("cos_k: K4V4 ~= K4V2 (same keys, |d|<0.02)",
          abs(g.loc["K4V4", "cos_k"] - g.loc["K4V2", "cos_k"]) < 0.02)
    check("cos_v: K4V4 > K4V2  (value bits drive cos_v)",
          g.loc["K4V4", "cos_v"] > g.loc["K4V2", "cos_v"] + 0.01)
    check("cos_v: K4V2 ~= K2V2 (same values, |d|<0.02)",
          abs(g.loc["K4V2", "cos_v"] - g.loc["K2V2", "cos_v"]) < 0.02)
    check("attn_js: K2V2 > K4V4 (worse keys shift attn)",
          g.loc["K2V2", "attn_js"] > g.loc["K4V4", "attn_js"])
    check("relmse_k: K2V2 > K4V4",
          g.loc["K2V2", "relmse_k"] > g.loc["K4V4", "relmse_k"])
    check("all cos in [0.5, 1]",
          ((g[["cos_k", "cos_v"]] >= 0.5) & (g[["cos_k", "cos_v"]] <= 1.0)).all().all())

    s = pd.read_csv(args.scores)
    check("both seeds in scores", {0, 1} <= set(s.seed.unique()))
    fp = s[s.config == "fp16"].wer.mean()
    check(f"fp16 WER < 0.20 (English preset; got {fp:.3f})", fp < 0.20)
    w = s.groupby("config").wer.mean()
    check("K2V2@0 WER >= fp16 WER", w.get("K2V2@0", 9) >= fp)
    print("\nper-config CER/WER:")
    print(s.groupby("config")[["cer", "wer"]].mean().round(3).to_string())

    print("\n===== SMOKE " + ("PASS — safe to launch the heavy Eddie batch"
          if ok else "FAIL — inspect above") + " =====")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
