"""Paired analysis of a 14_qwen_keyfix.sh run (plain / g32 / pre / g32pre cells).

Inputs (a downloaded archive, see docs/keyfix_experiment.md):
  <root>/results/qwen_keyfix_<cell>_scores.csv                 audio scores
  <root>/results/keyfix_reduced/qwen_keyfix_<cell>_seed*_bysent.csv
  <root>/results/keyfix_reduced/qwen_keyfix_<cell>_seed*_bypos.csv
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

CELLS = ["plain", "g32", "pre", "g32pre"]
CELL_LABEL = {"plain": "plain", "g32": "g32", "pre": "pre-RoPE", "g32pre": "g32 + pre-RoPE"}
# validated default categorical palette (dataviz skill), fixed order; markers = second encoding
CELL_STYLE = {"plain": ("#2a78d6", "o", "-"), "g32": ("#eb6834", "s", "-"),
              "pre": ("#1baf7a", "^", "--"), "g32pre": ("#eda100", "D", "--")}
METRICS = ["attn_js", "attn_top1", "out_cos", "cos_k", "relmse_k"]


def wmean(g, cols):
    w = g["n"]
    return pd.Series({c: float((g[c] * w).sum() / w.sum()) for c in cols} | {"n": int(w.sum())})


def load(root):
    scores, div, pos = {}, {}, {}
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
    return scores, div, pos


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
    scores, div, pos = load(args.root)
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
        for a, b in [("plain", "g32"), ("plain", "pre"), ("g32", "g32pre"), ("pre", "g32pre")]:
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

    # ---- 6. per-sentence paired divergence plain vs g32 --------------------
    if "plain" in div and "g32" in div:
        L += ["## 6. Paired per-sentence attn_js, plain vs g32 (talker, all layers)", "", "| config | plain mean | g32 mean | sentences improved | Wilcoxon p |", "|---|---|---|---|---|"]
        for (kb, vb) in sorted({(int(a), int(b)) for a, b in zip(div["plain"].key_bits, div["plain"].value_bits)}):
            ms = {}
            for c in ("plain", "g32"):
                d = div[c]; d = d[(d.key_bits == kb) & (d.value_bits == vb)]
                ms[c] = d.groupby("idx").apply(lambda g: wmean(g, ["attn_js"]), include_groups=False)["attn_js"]
            j = pd.concat(ms, axis=1).dropna()
            L.append(f"| {cfg_label(kb, vb)} | {j['plain'].mean():.4f} | {j['g32'].mean():.4f} | {int((j['g32'] < j['plain']).sum())}/{len(j)} | {wilcoxon(j['plain'], j['g32']).pvalue:.2e} |")
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

    with open(os.path.join(args.out, "keyfix_analysis.md"), "w") as f:
        f.write("\n".join(L))
    print("\n".join(L))

    # ---- figure ------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = [k for k in ("K2V2", "K3V3") if k in layer_tabs]
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
