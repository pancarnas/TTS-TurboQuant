"""Shrink the per-step divergence CSVs of 14_qwen_keyfix.sh into two small tables.

The raw files (results/qwen_keyfix_<cell>_seed<N>.csv) hold one row per
(sentence, layer, position, config) and reach ~750 MB each, which is too big
to download from the GPU box.  This writes, per raw file:

  <out>/<stem>_bysent.csv  mean of every metric per
                           (group, idx, layer, config, talker) with a row count
  <out>/<stem>_bypos.csv   talker-only mean per (layer, config, pos_bin)

`talker` is pos >= --talker-min-pos (33), the same split tools/summarize_keyfix.py
uses to drop code-predictor rows.  Both tables are exact means of the raw rows,
so any per-sentence / per-layer / per-position summary can be rebuilt from them
(weight by `n` when averaging across groups).

Runs in chunks, so memory stays flat regardless of file size.
"""
import argparse
import glob
import os

import pandas as pd

CFG = ["key_bits", "value_bits", "rw", "protected_layers", "key_group", "pre_rope"]
METRICS = ["attn_js", "attn_tv", "attn_top1", "attn_dentropy", "out_cos",
           "cos_k", "cos_v", "relmse_k", "relmse_v"]


def _accumulate(parts, chunk, keys, metrics):
    g = chunk.groupby(keys, sort=False)
    s = g[metrics].sum()
    s["n"] = g.size()
    parts.append(s.reset_index())


def _finish(parts, keys, metrics):
    if not parts:
        return pd.DataFrame(columns=keys + metrics + ["n"])
    df = pd.concat(parts, ignore_index=True).groupby(keys, sort=True).sum().reset_index()
    for m in metrics:
        df[m] = df[m] / df["n"]
    return df


def reduce_file(path, out_dir, min_pos, pos_bin, chunksize):
    stem = os.path.splitext(os.path.basename(path))[0]
    by_sent, by_pos = [], []
    cfg = None
    for chunk in pd.read_csv(path, chunksize=chunksize):
        if cfg is None:
            cfg = [c for c in CFG if c in chunk.columns]
            metrics = [m for m in METRICS if m in chunk.columns]
        chunk["talker"] = (chunk["pos"] >= min_pos).astype(int)
        _accumulate(by_sent, chunk, ["group", "idx", "layer"] + cfg + ["talker"], metrics)
        t = chunk[chunk["talker"] == 1].copy()
        if len(t):
            t["pos_bin"] = (t["pos"] // pos_bin) * pos_bin
            _accumulate(by_pos, t, ["layer"] + cfg + ["pos_bin"], metrics)
    if cfg is None:
        print(f"{path}: empty, skipped")
        return
    os.makedirs(out_dir, exist_ok=True)
    a = _finish(by_sent, ["group", "idx", "layer"] + cfg + ["talker"], metrics)
    b = _finish(by_pos, ["layer"] + cfg + ["pos_bin"], metrics)
    a.to_csv(os.path.join(out_dir, f"{stem}_bysent.csv"), index=False)
    b.to_csv(os.path.join(out_dir, f"{stem}_bypos.csv"), index=False)
    print(f"{path}: {int(a['n'].sum())} raw rows -> {len(a)} by-sentence rows, {len(b)} by-position rows")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--pattern", default="qwen_keyfix_*_seed*.csv")
    ap.add_argument("--out", default="results/keyfix_reduced")
    ap.add_argument("--talker-min-pos", type=int, default=33)
    ap.add_argument("--pos-bin", type=int, default=8)
    ap.add_argument("--chunksize", type=int, default=2_000_000)
    args = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(args.results, args.pattern)))
    if not paths:
        raise SystemExit(f"no files match {args.results}/{args.pattern}")
    for p in paths:
        reduce_file(p, args.out, args.talker_min_pos, args.pos_bin, args.chunksize)


if __name__ == "__main__":
    main()
