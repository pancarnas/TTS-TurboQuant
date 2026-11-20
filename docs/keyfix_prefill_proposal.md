# Prefill rescue after the prefix-codebook run: what to try next

Date 2026-11-20. Branch `exp/keyfix-group-prerope`. Follows the pk2 / pk3 / pm3
run of 2026-11-19 (summary in `results/keyfix_summary.md` on the box). Run book
`docs/keyfix_experiment.md`, design of the codebook cells
`docs/superpowers/specs/2026-11-18-prefix-codebook-design.md`.

## What the data says

K2V2@0, 50 LibriSpeech-PC sentences, seed 0, generated keys on g32pre in every
new cell:

| cell | prompt keys | prompt values | collapse | CER | spk_sim | attn_js | relmse_k |
|---|---|---|---|---|---|---|---|
| g32pre | 2-bit fixed codebook, g32 | 2-bit fixed | 100 % | 1.04 | 0.75 | 0.061 | 0.016 |
| pk2 | k-means levels, 2 bit | 2-bit fixed | 100 % | 0.99 | 0.68 | 0.065 | 0.011 |
| pk3 | k-means levels, 3 bit | 2-bit fixed | 100 % | 0.97 | 0.68 | 0.033 | 0.005 |
| pm3 | one min-max window, 3 bit | 2-bit fixed | 36 % | 0.33 | 0.92 | 0.017 | 0.006 |
| g32pp | fp16 | fp16 | 0 % | 0.003 | 0.99 | 0.019 | 0.015 |

Three facts to build on.

1. **MSE on the prompt keys is not the target.** pk3 has the lowest key error
   of all nine cells and collapses every sentence; pm3 has a worse error and
   collapses a third. The two quantizers differ only in where the levels sit:
   min-max keeps each coordinate's minimum and maximum as exact levels,
   k-means places levels at the quantile centres and shrinks the extremes.
   Attention logits are inner products, dominated by the few large-magnitude
   coordinates of the few heavily attended keys. Whatever the fix is, it must
   preserve those extremes, not the average.
2. **Prompt values are a live confound.** All three codebook cells quantize
   the prompt values with the ordinary 2-bit codebook (cos_v is 0.9406 in all
   of them); g32pp keeps them fp16. pm3's remaining 36 % may be values, not
   keys. Nothing in the run separates the two.
3. **Mean divergence still does not predict collapse.** pm3's attention JS is
   below g32pp's. Collapse is a threshold event on something concentrated,
   which again points at a few keys rather than at all of them.

The keygauss diagnostic (handoff, "Gaussianity diagnostic") adds: the prompt
keys do not drift between windows (drift -0.04), carry 23-73 % of the
attention mass per layer, and the per-step attention mass on the first 64 keys
and on the prompt is already recorded in `results/keygauss_stats*`.

## What the literature does about exactly this

The failure mode, "a few prompt tokens or channels with extreme keys carry the
attention and break under uniform quantization", has three established fixes.

- **Keep the pivot or sink tokens exact.** IntactKV ([Liu et al., ACL Findings
  2024](https://aclanthology.org/2024.findings-acl.460.pdf)) keeps the KV of the
  [BOS] and a few pivot tokens in full precision and reports that this alone
  removes most of the quantization damage in self-attention, at no inference
  cost. KVSink ([arXiv 2508.04257](https://arxiv.org/html/2508.04257v1))
  predicts the sink tokens from stable outliers; Outlier Tokens Tracing
  ([arXiv 2505.10938](https://arxiv.org/pdf/2505.10938)) picks them
  dynamically, with a full-precision recent window alongside, which is our
  `residual_window`. InnerQ ([arXiv 2602.23200](https://arxiv.org/pdf/2602.23200v2))
  reserves part of the high-precision window for the first tokens so sink
  outliers do not leak into the quantization groups.
- **Isolate the outlier coordinates, not the tokens.** KVQuant ([Hooper et
  al., NeurIPS 2024](https://arxiv.org/abs/2401.18079)) quantizes keys
  per-channel and pre-RoPE (what g32pre already does), and keeps about 1 % of
  the largest-magnitude entries per vector in a sparse fp16 residual
  ("dense-and-sparse"). They report under 0.1 perplexity loss at 3 bits with
  that 1 %. Their non-uniform datatype is a *sensitivity-weighted* k-means,
  which is the version of our k-means that would not have shrunk the tails.
- **Spend bits where the queries look.** MixKVQ ([ACL 2026](https://aclanthology.org/2026.acl-long.326.pdf))
  uses query activations to find the key channels that need more precision;
  "Spend Bits Where Queries Look" ([arXiv 2608.04074](https://arxiv.org/html/2608.04074))
  quantizes the prompt at prefill with attention-product error as the
  distortion measure rather than MSE, which is the formal version of fact 1.

The first family costs nothing in code (we have `--protect-prefix N`), the
second is about forty lines in the compressor, the third is a redesign of the
level fitting and is not where the evidence points first.

## Proposal: four cells, cheapest first, with a decision after each pair

All at `CFGS="fp16,K2V2@0" MAXPG=50`, about 2.5 GPU hours per cell on the L4.
Prompt bit costs are per prompt key element, norm excluded, P = 53 (as in
section 8 of the analysis).

### Pair 1: zero code, run back to back (about 5 h)

**A. pm3 with 4-bit values**, `CELLS="pm3" CFGS="fp16,K2V4@0"`. Same cell, same
prompt keys, prompt values at 4 bits. Isolates fact 2. K2V4 on g32pre already
exists in the summary (100 % collapse), so the control is in hand.

**B. g32pre plus the first 4 prompt tokens in fp16**, a `g32prepp4` cell:
`--key-group 32 --pre-rope-keys --protect-prefix 4`. One line in
`cell_flags` of `14_qwen_keyfix.sh` and the cell names in
`tools/summarize_keyfix.py` and `tools/analyze_keyfix.py`; the wav suffix
`_pp=4`, the scorer and the resume filter already handle fixed N. This is the
IntactKV / KVSink test: if the sink carries it, 4 exact tokens cost 1.2 bits
per prompt key element (64 / 53) and nothing else changes. It is also item 2c
of the handoff, so it was due anyway.

Decision:

| A | B | reading | next |
|---|---|---|---|
| ~0 % | any | the 2-bit prompt **values** were the carrier | candidate = pm3 keys + V3 or V4 values; run `pm3` at `K2V3@0` to find the floor |
| ~36 % | ~0 % | an attention-**sink** effect, 4 tokens | candidate = 4 fp16 tokens + 2-bit everything else; sweep N = 2, 8 to find the smallest N |
| ~36 % | ~36 % or worse | neither; the **extremes across the whole prompt** are the carrier | Pair 2 |
| ~0 % | ~0 % | both help | combine: pp4 + pm3 keys + V4 values, see Pair 2 D |

### Pair 2: small code, only if Pair 1 does not settle it

**C. pm4**, min-max prompt at 4 bits. One line each in the run script and the
tools' cell lists; the library already accepts `--prefix-bits 4`. Prompt cost
4.6 bits per element. If this reaches 0 %, the thesis has a fully quantized
cache at a stated cost and the story is "range, not shape". If 36 % does not
move, bits are not the lever and D is.

**D. Dense-and-sparse prompt keys** (KVQuant): quantize the prompt keys with
the min-max window at 3 bits and keep the top 1 % largest-magnitude entries of
each prompt key vector in fp16 as a sparse residual (index + value). About
forty lines in `MSECompressor` (a `sparse_frac` on the levels stage: mask the
outliers out before fitting, store them, add them back after reconstruct),
one `TurboQuantConfig` field, one cell. Cost 1 % x (16 + 7 index) = 0.23 bits
per element on top of pm3's 3.6. This is the direct test of fact 1: it keeps
the tails exact and nothing else. If D rescues where C does not, the thesis
claim is "the collapse lives in a few hundred extreme prompt-key entries".

### Not proposed, and why

- More k-means variants (weighted, more iterations, better init): fact 1 says
  the objective is wrong, not the optimizer. The sensitivity-weighted version
  is D in disguise with more code.
- The QJL / asymmetric estimator in `turboquant/compressors.py`: it fixes
  inner-product bias, which is the right idea, but it needs attention computed
  from the compressed form and the Qwen cache writes reconstructed tensors
  back into the buffer. That is a different cache, not a cell.
- Query-aware level fitting at prefill: the prefill queries are available, but
  the generated queries are what attend to the prompt later. A separate
  experiment after the four cells, if the thesis needs it.

## What each cell must report

Same protocol as every cell: collapse count out of 50, CER, WER, spk_sim; the
talker-row attention and key metrics; and, after the reducer, analysis section
8 (effective bits with the prompt cost), 9 (runaway versus early stop among
the collapses) and 10 (per-step maximum against collapse). The honest headline
for each candidate is "collapse X / 50 at Y bits per key element pooled over
the sentence", next to g32pp's 0 / 50 at about 5 bits.

## Commands

Pair 1 A, now, no code:

```bash
CELLS="pm3" CFGS="fp16,K2V4@0" MAXPG=50 nohup bash scripts/gpu/14_qwen_keyfix.sh > logs/keyfix_pm3_v4.log 2>&1 &
```

Pair 1 B needs the `g32prepp4` line committed first; then:

```bash
CELLS="g32prepp4" CFGS="fp16,K2V2@0" MAXPG=50 nohup bash scripts/gpu/14_qwen_keyfix.sh > logs/keyfix_pp4.log 2>&1 &
```

Both resume per sentence. Reduce and download after each pair, as in the run
book.
