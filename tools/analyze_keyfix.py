"""Paired analysis of a 14_qwen_keyfix.sh run (plain / g32 / pre / g32pre / pp / g32pp cells).

Inputs (a downloaded archive, see docs/keyfix_experiment.md):
  <root>/results/qwen_keyfix_<cell>_scores.csv                 audio scores
  <root>/results/keyfix_reduced/qwen_keyfix_<cell>_seed*_bysent.csv
  <root>/results/keyfix_reduced/qwen_keyfix_<cell>_seed*_bypos.csv
  <root>/results/keyfix_reduced/qwen_keyfix_<cell>_seed*_bylen.csv   (optional, bit accounting)
      (from tools/reduce_keyfix_div.py; talker-only rows are talker == 1)

Outputs:
  <out>/keyfix_analysis.md    paired tables + tests
  <out>/keyfix_layers.{png,pdf}  talker-only per-layer attn_js, one panel per config

All sentences are shared across cells (same idx, same seed), so every
comparison below is paired per sentence.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest, mannwhitneyu, wilcoxon

CELLS = ["plain", "g32", "pre", "g32pre", "pp", "g32pp", "pk2", "pk3", "pm3", "pm4"]
CELL_LABEL = {"plain": "plain", "g32": "g32", "pre": "pre-RoPE", "g32pre": "g32 + pre-RoPE",
              "pp": "prompt fp16", "g32pp": "g32 + prompt fp16", "pk2": "kmeans prompt 2b + g32pre",
              "pk3": "kmeans prompt 3b + g32pre", "pm3": "minmax prompt 3b + g32pre",
              "pm4": "minmax prompt 4b + g32pre"}
# fixed categorical order; markers and line styles are the second encoding
CELL_STYLE = {"plain": ("#2a78d6", "o", "-"), "g32": ("#eb6834", "s", "-"),
              "pre": ("#1baf7a", "^", "--"), "g32pre": ("#eda100", "D", "--"),
              "pp": ("#8a5cf5", "v", ":"), "g32pp": ("#c8398a", "P", "-."),
              "pk2": ("#0f9bb0", "X", "-"), "pk3": ("#6b8e23", "<", "--"), "pm3": ("#a0522d", ">", ":"),
              "pm4": ("#444444", "*", "-.")}
PAIRS = [("plain", "g32"), ("plain", "pre"), ("plain", "pp"), ("g32", "g32pre"), ("pre", "g32pre"),
         ("g32", "g32pp"), ("pp", "g32pp"), ("g32pre", "g32pp")]
PAIRS += [(a, c) for c in ("pk2", "pk3", "pm3", "pm4") for a in ("g32pre", "g32pp")]
METRICS = ["attn_js", "attn_top1", "out_cos", "cos_k", "relmse_k"]


def wmean(g, cols):
    w = g["n"]
    return pd.Series({c: float((g[c] * w).sum() / w.sum()) for c in cols} | {"n": int(w.sum())})


def load(root):
    scores, div, pos, lens, steps = {}, {}, {}, {}, {}
    for c in CELLS:
        p = os.path.join(root, "results", f"qwen_keyfix_{c}_scores.csv")
        if not os.path.exists(p):
            continue
        s = pd.read_csv(p)
        s["base"] = s["config"].str.split("+").str[0]
        s["collapse"] = (s["cer"] > 0.5).astype(int)
        scores[c] = s
        bs = sorted(glob.glob(os.path.join(root, "results", "keyfix_reduced", f"qwen_keyfix_{c}_seed*_bysent.csv")))
        bp = sorted(glob.glob(os.path.join(root, "results", "keyfix_reduced", f"qwen_keyfix_{c}_seed*_bypos.csv")))
        if bs:
            div[c] = pd.concat([pd.read_csv(x) for x in bs], ignore_index=True)
            div[c] = div[c][div[c]["talker"] == 1]
        if bp:
            pos[c] = pd.concat([pd.read_csv(x) for x in bp], ignore_index=True)
        bl = sorted(glob.glob(os.path.join(root, "results", "keyfix_reduced", f"qwen_keyfix_{c}_seed*_bylen.csv")))
        if bl:
            lens[c] = pd.concat([pd.read_csv(x) for x in bl], ignore_index=True)
            lens[c]["prefix_bits"] = lens[c]["prefix_bits"].fillna(0) if "prefix_bits" in lens[c] else 0
            lens[c]["prefix_codebook"] = lens[c]["prefix_codebook"].fillna("") if "prefix_codebook" in lens[c] else ""
        bt = sorted(glob.glob(os.path.join(root, "results", "keyfix_reduced", f"qwen_keyfix_{c}_seed*_bystep.csv")))
        if bt:
            steps[c] = pd.concat([pd.read_csv(x) for x in bt], ignore_index=True)
    return scores, div, pos, lens, steps


def effective_bits(bits, meta_bits, prefix_fp16, P, S, prefix_bits_per_elem=None):
    """Pooled bits per element over sentences: fp16 (or prefix_bits_per_elem when given) for the protected
    prefix, codes + metadata for the rest. Ignores the per-vector fp16 norm (same in every cell)."""
    P = np.minimum(P, S) if prefix_fp16 else np.zeros_like(S)
    pcost = 16.0 if prefix_bits_per_elem is None else prefix_bits_per_elem
    return float((pcost * P.sum() + (bits + meta_bits) * (S - P).sum()) / S.sum())


def prefix_cell_bits(g, kb, vb):
    """Pooled key/value bits for one (cell, config) frame; codebook cells cost pb bits per prompt element
    (keys add the fitted levels' fp16 scalars; prompt values are quantized at vb by the value compressor)."""
    P, S = g["prompt_len"].to_numpy().astype(float), g["seq_len"].to_numpy().astype(float)
    pp = int(g["protect_prefix"].fillna(0).iloc[0]) if "protect_prefix" in g else 0
    Pp = P if pp < 0 else np.minimum(P, float(pp))
    meta = 1.0 if int(g["key_group"].iloc[0]) > 0 else 0.0
    mode, pb = str(g["prefix_codebook"].iloc[0]), int(g["prefix_bits"].iloc[0]) or kb
    cost = vcost = None
    if mode:  # codebook-coded prompt: pb bits per element plus the fitted levels' fp16 scalars
        from turboquant.fitted_levels import meta_scalars
        cost, vcost = pb + meta_scalars(pb, mode) * 16 / Pp.mean(), float(vb)
    S1000 = np.full_like(S, 1000.0)
    return {"key": effective_bits(kb, meta, pp != 0, Pp, S, cost),
            "value": effective_bits(vb, 0.0, pp != 0, Pp, S, vcost),
            "key_1000": effective_bits(kb, meta, pp != 0, Pp, S1000, cost), "mode": mode, "pb": pb}


def runaway_table(scores):
    """Per (cell, base config): collapses (CER > 0.5), runaways (collapsed, duration > 3x the sentence's fp16
    duration), early stops (< 1.5x), and the median ratio / speaker similarity over the collapsed sentences."""
    rows = []
    for cell, s in scores.items():
        ref = s[s["config"] == "fp16"].drop_duplicates("idx").set_index("idx")["dur_s"]
        q = s[s["config"] != "fp16"].copy()
        q["base"] = q["config"].str.split("+").str[0]
        q["ratio"] = q["dur_s"] / q["idx"].map(ref)
        for base, g in q.groupby("base"):
            c = g[g["cer"] > 0.5]
            rows.append({"cell": cell, "config": base, "n_collapse": len(c), "runaway": int((c["ratio"] > 3).sum()),
                         "stopped": int((c["ratio"] < 1.5).sum()), "ratio_median": c["ratio"].median(),
                         "spk_median": c["spk_sim"].median()})
    return pd.DataFrame(rows, columns=["cell", "config", "n_collapse", "runaway", "stopped", "ratio_median",
                                       "spk_median"])


def _auc_p(stat, bad):
    """P(collapsed > survived) (ties 0.5) and the Mann-Whitney p of a per-sentence statistic."""
    a, b = stat[stat.index.isin(bad[bad].index)], stat[stat.index.isin(bad[~bad].index)]
    if not len(a) or not len(b):
        return float("nan"), float("nan")
    r = mannwhitneyu(a, b)
    return float(r.statistic / (len(a) * len(b))), float(r.pvalue)


def step_table(div_step, scores):
    """Per (cell, config) with both collapsed and surviving sentences: AUC and Mann-Whitney p of the
    per-sentence max over layers of js_max, of js_p95, and of layer-5 js_max (from the _bystep tables)."""
    rows = []
    for cell, st in div_step.items():
        if cell not in scores:
            continue
        st = st.assign(config=[f"K{int(k)}V{int(v)}@{int(r)}" for k, v, r in zip(st.key_bits, st.value_bits, st.rw)])
        sc = scores[cell].assign(base=lambda d: d["config"].str.split("+").str[0])
        for cfg, g in st.groupby("config"):
            bad = (sc[sc["base"] == cfg].drop_duplicates("idx").set_index("idx")["cer"] > 0.5)
            if bad.all() or not bad.any():
                continue
            stats = {"js_max": g.groupby("idx")["js_max"].max(), "js_p95": g.groupby("idx")["js_p95"].max(),
                     "l5_max": g[g["layer"] == 5].groupby("idx")["js_max"].max()}
            row = {"cell": cell, "config": cfg, "n_collapse": int(bad.sum()), "n_ok": int((~bad).sum())}
            for k, v in stats.items():
                row[f"auc_{k}"], row[f"p_{k}"] = _auc_p(v, bad)
            rows.append(row)
    return pd.DataFrame(rows)


def cfg_label(kb, vb):
    return f"K{kb}V{vb}"


def md_table(df, floatfmt="{:.3f}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)) and c in ("idx", "layer") and float(v).is_integer():
                v = int(v)
            cells.append(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="results/keyfix")
    ap.add_argument("--out", default="results/keyfix")
    args = ap.parse_args()
    scores, div, pos, lens, steps = load(args.root)
    os.makedirs(args.out, exist_ok=True)
    L = []

    # ---- 1. audio, per cell ------------------------------------------------
    S = pd.concat([s.assign(cell=c) for c, s in scores.items()], ignore_index=True)
    S = S[S["base"] != "fp16"]
    t = (S.groupby(["base", "cell"]).agg(n=("cer", "size"), cer=("cer", "mean"), cer_median=("cer", "median"),
                                          collapse_pct=("collapse", lambda x: 100 * x.mean()), spk_sim=("spk_sim", "mean"))
         .reset_index())
    t["cell"] = pd.Categorical(t["cell"], CELLS)
    t = t.sort_values(["base", "cell"])
    L += ["# Key-fix paired analysis (Qwen3-TTS, clone mode, seed 0)", "",
          "Same 50 sentences and seed in every cell; collapse = CER > 0.5.", "",
          "## 1. Audio per cell", "", md_table(t), ""]

    # ---- 2. paired collapse tests vs plain and g32 vs g32pre --------------
    L += ["## 2. Paired collapse tests (per sentence)", "",
          "McNemar-style exact test on discordant sentences: b = collapsed only in A, c = collapsed only in B.", "",
          "| config | A | B | both | A only | B only | neither | p (two-sided binomial on A-only vs B-only) |", "|---|---|---|---|---|---|---|---|"]
    for base in sorted(S["base"].unique()):
        P = S[S["base"] == base].pivot(index="idx", columns="cell", values="collapse")
        for a, b in PAIRS:
            if a not in P or b not in P:
                continue
            both = int(((P[a] == 1) & (P[b] == 1)).sum()); ao = int(((P[a] == 1) & (P[b] == 0)).sum())
            bo = int(((P[a] == 0) & (P[b] == 1)).sum()); nei = int(((P[a] == 0) & (P[b] == 0)).sum())
            p = binomtest(ao, ao + bo, 0.5).pvalue if ao + bo else float("nan")
            L.append(f"| {base} | {a} | {b} | {both} | {ao} | {bo} | {nei} | {p:.3f} |")
    L.append("")

    # ---- 3. per-sentence CER table at K3V3 --------------------------------
    k3 = S[S["base"] == "K3V3@0"]
    if len(k3):
        C = k3.pivot(index="idx", columns="cell", values="cer")[[c for c in CELLS if c in k3["cell"].unique()]]
        L += ["## 3. K3V3 CER per sentence", "", md_table(C.reset_index(), "{:.2f}"), ""]

    # ---- 4. divergence per layer, per config ------------------------------
    layer_tabs = {}
    for (kb, vb) in sorted({(int(a), int(b)) for d in div.values() for a, b in zip(d.key_bits, d.value_bits)}):
        tab = {}
        for c, d in div.items():
            x = d[(d.key_bits == kb) & (d.value_bits == vb)]
            if len(x):
                tab[c] = x.groupby("layer").apply(lambda g: wmean(g, ["attn_js"]), include_groups=False)["attn_js"]
        T = pd.DataFrame(tab)
        layer_tabs[cfg_label(kb, vb)] = T
        T2 = T.copy()
        if "plain" in T2 and "g32" in T2:
            T2["plain/g32"] = T2["plain"] / T2["g32"]
        L += [f"## 4. Talker-only per-layer attn_js, {cfg_label(kb, vb)}", "", md_table(T2.reset_index(), "{:.4f}"), "",
              "Peak layer: " + ", ".join(f"{c}: {int(T[c].idxmax())}" for c in T.columns), ""]

    # ---- 5. does divergence predict collapse within a cell? ---------------
    L += ["## 5. Per-sentence divergence vs collapse (within cell, K3V3)", "",
          "Mean talker attn_js over all layers per sentence, split by whether that sentence collapsed.", "",
          "| cell | n collapsed | attn_js collapsed | attn_js ok | layer-5 collapsed | layer-5 ok | Mann-Whitney p |", "|---|---|---|---|---|---|---|"]
    for c in CELLS:
        if c not in div or c not in scores:
            continue
        d = div[c]; d = d[(d.key_bits == 3) & (d.value_bits == 3)]
        s = scores[c]; s = s[(s.key_bits == 3) & (s.value_bits == 3)].set_index("idx")["collapse"]
        if not len(d) or s.sum() in (0, len(s)):
            m = d.groupby("idx").apply(lambda g: wmean(g, ["attn_js"]), include_groups=False)["attn_js"]
            L.append(f"| {c} | {int(s.sum())} | {m[s[s == 1].index].mean() if s.sum() else float('nan'):.4f} | "
                     f"{m[s[s == 0].index].mean() if (s == 0).sum() else float('nan'):.4f} | | | (no split) |")
            continue
        m = d.groupby("idx").apply(lambda g: wmean(g, ["attn_js"]), include_groups=False)["attn_js"]
        m5 = d[d.layer == 5].groupby("idx").apply(lambda g: wmean(g, ["attn_js"]), include_groups=False)["attn_js"]
        a, b = m[s[s == 1].index], m[s[s == 0].index]
        p = mannwhitneyu(a, b).pvalue
        L.append(f"| {c} | {int(s.sum())} | {a.mean():.4f} | {b.mean():.4f} | {m5[s[s == 1].index].mean():.4f} | {m5[s[s == 0].index].mean():.4f} | {p:.3f} |")
    L.append("")

    # ---- 6. per-sentence paired divergence between cells ------------------
    L += ["## 6. Paired per-sentence attn_js between cells (talker, all layers)", "",
          "| pair | config | A mean | B mean | sentences improved | Wilcoxon p |", "|---|---|---|---|---|---|"]
    for a, b in [("plain", "g32"), ("plain", "pp"), ("g32", "g32pp"), ("pp", "g32pp"), ("g32pre", "g32pp")]:
        if a not in div or b not in div:
            continue
        cfgs = sorted({(int(x), int(y)) for x, y in zip(div[a].key_bits, div[a].value_bits)}
                      & {(int(x), int(y)) for x, y in zip(div[b].key_bits, div[b].value_bits)})
        for (kb, vb) in cfgs:
            ms = {}
            for c in (a, b):
                d = div[c]; d = d[(d.key_bits == kb) & (d.value_bits == vb)]
                ms[c] = d.groupby("idx").apply(lambda g: wmean(g, ["attn_js"]), include_groups=False)["attn_js"]
            j = pd.concat(ms, axis=1).dropna()
            if len(j) < 5:
                continue
            L.append(f"| {a} vs {b} | {cfg_label(kb, vb)} | {j[a].mean():.4f} | {j[b].mean():.4f} | "
                     f"{int((j[b] < j[a]).sum())}/{len(j)} | {wilcoxon(j[a], j[b]).pvalue:.2e} |")
    L.append("")

    # ---- 7. divergence along the sequence ---------------------------------
    if pos:
        edges = [0, 64, 128, 256, 512, 1024, 100000]
        names = ["<64", "64-127", "128-255", "256-511", "512-1023", ">=1024"]
        L += ["## 7. attn_js along the generated sequence (K3V3, talker)", "",
              "Mean per position range of the decode step; plain grows with length, g32 does not.", "",
              "| cell | layer | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
        for c in CELLS:
            if c not in pos:
                continue
            d = pos[c]; d = d[(d.key_bits == 3) & (d.value_bits == 3)].copy()
            d["rng"] = pd.cut(d["pos_bin"], edges, right=False, labels=names)
            for layer in (0, 5, 15):
                x = d[d.layer == layer]
                g = x.groupby("rng", observed=True).apply(lambda g: wmean(g, ["attn_js"]), include_groups=False)["attn_js"]
                L.append(f"| {c} | {layer} | " + " | ".join(f"{g.get(n, float('nan')):.4f}" for n in names) + " |")
        L.append("")

    # ---- 8. effective bits per element (prompt protection + group metadata) --
    if lens:
        L += ["## 8. Effective bits per cache element (codes + group metadata + fp16 prefix; per-vector norm excluded)", "",
              "Pooled over the 50 sentences: (16 P + b_eff (S - P)) / S with P = protected prompt keys, S = all keys. "
              "'at S=1000' re-evaluates the same formula for a 1000-key sequence with the same prompts.", "",
              "| cell | config | prompt P (min/mean/max) | S (min/mean/max) | key bits | value bits | key bits at S=1000 |",
              "|---|---|---|---|---|---|---|"]
        for c in CELLS:
            if c not in lens:
                continue
            d = lens[c]
            for (kb, vb), g in d.groupby(["key_bits", "value_bits"]):
                P, S = g["prompt_len"].to_numpy().astype(float), g["seq_len"].to_numpy().astype(float)
                bits = prefix_cell_bits(g, int(kb), int(vb))
                kbits, vbits, k1000, mode, pb = (bits[k] for k in ("key", "value", "key_1000", "mode", "pb"))
                label = cfg_label(int(kb), int(vb)) + (f" {mode} {pb}b prompt" if mode else "")
                L.append(f"| {c} | {label} | {int(P.min())}/{P.mean():.0f}/{int(P.max())} | "
                         f"{int(S.min())}/{S.mean():.0f}/{int(S.max())} | {kbits:.2f} | {vbits:.2f} | {k1000:.2f} |")
        L.append("")

    # ---- 9. runaway classification ----------------------------------------
    L += ["## 9. Runaway classification", "",
          "Collapse = CER > 0.5; runaway = collapsed with duration > 3x that sentence's fp16 duration; "
          "stopped = collapsed with duration < 1.5x. Medians are over the collapsed sentences.", "",
          md_table(runaway_table(scores)), ""]

    # ---- 10. per-step maximum vs collapse ---------------------------------
    L += ["## 10. Per-step maximum vs collapse", "",
          "Within a cell and config: AUC = P(collapsed sentence has the larger statistic), ties 0.5, with the "
          "Mann-Whitney p. Statistics per sentence: max over layers of the per-step js_max / js_p95, and layer-5 js_max.", ""]
    st = step_table(steps, scores)
    L += [md_table(st), ""] if len(st) else ["_no cell has both collapsed and surviving sentences with _bystep data_", ""]

    with open(os.path.join(args.out, "keyfix_analysis.md"), "w") as f:
        f.write("\n".join(L))
    print("\n".join(L))

    # ---- figure ------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = [k for k in ("K2V2", "K3V3") if k in layer_tabs]
    if not keys:
        print("no divergence tables, figure skipped")
        return
    fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 3.4), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, keys):
        T = layer_tabs[k]
        for c in CELLS:
            if c not in T:
                continue
            col, mk, ls = CELL_STYLE[c]
            ax.plot(T.index, T[c], color=col, marker=mk, ls=ls, lw=1.6, ms=4, mfc="white", mew=1.2, label=CELL_LABEL[c])
        ax.set_title(f"{k} keys/values, talker layers", fontsize=10)
        ax.set_xlabel("layer")
        ax.set_ylabel("attention JS divergence")
        ax.set_yscale("log")
        ax.grid(True, which="major", color="#e4e3df", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xticks(range(0, 28, 3))
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.out, f"keyfix_layers.{ext}"), dpi=200)
    print("figure ->", os.path.join(args.out, "keyfix_layers.png"))


if __name__ == "__main__":
    main()
