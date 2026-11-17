"""Shrink the per-step divergence CSVs of 14_qwen_keyfix.sh into two small tables.

The raw files (results/qwen_keyfix_<cell>_seed<N>.csv) hold one row per
(sentence, layer, position, config) and reach ~750 MB each, which is too big
to download from the GPU box.  This writes, per raw file:

  <out>/<stem>_bysent.csv  mean of every metric per
                           (group, idx, layer, config, talker) with a row count
  <out>/<stem>_bypos.csv   talker-only mean per (layer, config, pos_bin)
  <out>/<stem>_bystep.csv  talker-only per (group, idx, layer, config): max of
                           attn_js over steps, its 95th percentile (0.005 bins),
                           and the fraction of steps with attn_js > 0.1 / 0.5
  <out>/<stem>_bytime.csv  talker-only per (group, idx, config, pos_bin): mean
                           and max of attn_js over layers and steps in the bin
                           (the per-sentence time course)
  <out>/<stem>_bylen.csv   per (group, idx, config): prompt_len (pos of the
                           talker prefill row = number of prompt keys), seq_len
                           (last pos = prompt + generated keys) and the number
                           of recorded talker steps; needed for the effective
                           bits of the protect_prefix cells

`talker` is pos >= --talker-min-pos (33), the same split tools/summarize_keyfix.py
uses to drop code-predictor rows.  Both tables are exact means of the raw rows,
so any per-sentence / per-layer / per-position summary can be rebuilt from them
(weight by `n` when averaging across groups).

Runs in chunks, so memory stays flat regardless of file size.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

CFG = ["key_bits", "value_bits", "rw", "protected_layers", "key_group", "pre_rope", "protect_prefix"]
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


N_BINS = 200  # attn_js histogram bins on [0, 1] -> 0.005 resolution for the percentile


def _finish_steps(hist_parts, max_parts, keys):
    """Per key: exact max, histogram-based p95, fraction of steps above thresholds."""
    h = pd.concat(hist_parts, ignore_index=True).groupby(keys + ["bin"], sort=True)["cnt"].sum().reset_index()
    mx = pd.concat(max_parts, ignore_index=True).groupby(keys, sort=True)["js_max"].max()
    out = []
    for key, g in h.groupby(keys, sort=True):
        g = g.sort_values("bin")
        cnt = g["cnt"].to_numpy()
        edges = g["bin"].to_numpy()
        total = cnt.sum()
        cum = np.cumsum(cnt)
        p95_bin = edges[np.searchsorted(cum, 0.95 * total)]
        row = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        row.update(js_p95=(p95_bin + 1) / N_BINS, n=int(total),
                   js_frac_gt_0p1=float(cnt[edges >= 0.1 * N_BINS].sum() / total),
                   js_frac_gt_0p5=float(cnt[edges >= 0.5 * N_BINS].sum() / total))
        out.append(row)
    df = pd.DataFrame(out)
    df["js_max"] = mx.reindex(pd.MultiIndex.from_frame(df[keys]) if len(keys) > 1 else df[keys[0]]).to_numpy()
    return df[keys + ["js_max", "js_p95", "js_frac_gt_0p1", "js_frac_gt_0p5", "n"]]


def _finish_time(parts, keys):
    df = pd.concat(parts, ignore_index=True).groupby(keys, sort=True).agg(
        js_sum=("js_sum", "sum"), js_max=("js_max", "max"), top1_sum=("top1_sum", "sum"), n=("n", "sum")).reset_index()
    df["attn_js"] = df["js_sum"] / df["n"]
    df["attn_top1"] = df["top1_sum"] / df["n"]
    return df[keys + ["attn_js", "js_max", "attn_top1", "n"]]


def reduce_file(path, out_dir, min_pos, pos_bin, chunksize):
    stem = os.path.splitext(os.path.basename(path))[0]
    by_sent, by_pos, hist, mx, by_time, by_len = [], [], [], [], [], []
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
            skeys = ["group", "idx", "layer"] + cfg
            t["bin"] = np.clip(np.floor(t["attn_js"].to_numpy() * N_BINS), 0, N_BINS - 1).astype(int)
            hist.append(t.groupby(skeys + ["bin"], sort=False).size().rename("cnt").reset_index())
            mx.append(t.groupby(skeys, sort=False)["attn_js"].max().rename("js_max").reset_index())
            tkeys = ["group", "idx"] + cfg + ["pos_bin"]
            g = t.groupby(tkeys, sort=False)
            tt = g["attn_js"].sum().rename("js_sum").reset_index()
            tt["js_max"] = g["attn_js"].max().to_numpy()
            tt["top1_sum"] = g["attn_top1"].sum().to_numpy() if "attn_top1" in t.columns else 0.0
            tt["n"] = g.size().to_numpy()
            by_time.append(tt)
            lkeys = ["group", "idx"] + cfg
            gl = t.groupby(lkeys, sort=False)["pos"]
            ll = gl.min().rename("prompt_len").reset_index()
            ll["seq_len"] = gl.max().to_numpy()
            ll["n_steps"] = gl.size().to_numpy()
            by_len.append(ll)
    if cfg is None:
        print(f"{path}: empty, skipped")
        return
    os.makedirs(out_dir, exist_ok=True)
    a = _finish(by_sent, ["group", "idx", "layer"] + cfg + ["talker"], metrics)
    b = _finish(by_pos, ["layer"] + cfg + ["pos_bin"], metrics)
    a.to_csv(os.path.join(out_dir, f"{stem}_bysent.csv"), index=False)
    b.to_csv(os.path.join(out_dir, f"{stem}_bypos.csv"), index=False)
    msg = f"{path}: {int(a['n'].sum())} raw rows -> {len(a)} by-sentence rows, {len(b)} by-position rows"
    if hist:
        c = _finish_steps(hist, mx, ["group", "idx", "layer"] + cfg)
        d = _finish_time(by_time, ["group", "idx"] + cfg + ["pos_bin"])
        c.to_csv(os.path.join(out_dir, f"{stem}_bystep.csv"), index=False)
        d.to_csv(os.path.join(out_dir, f"{stem}_bytime.csv"), index=False)
        lk = ["group", "idx"] + cfg
        e = (pd.concat(by_len, ignore_index=True).groupby(lk, sort=True)
             .agg(prompt_len=("prompt_len", "min"), seq_len=("seq_len", "max"), n_steps=("n_steps", "sum")).reset_index())
        e.to_csv(os.path.join(out_dir, f"{stem}_bylen.csv"), index=False)
        msg += f", {len(c)} by-step rows, {len(d)} by-time rows, {len(e)} by-length rows"
    print(msg)


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
