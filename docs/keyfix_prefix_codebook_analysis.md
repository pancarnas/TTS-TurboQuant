# Prefix codebook experiment: what it did and what it found

Date 2026-11-23. Branch `exp/keyfix-group-prerope`. Covers the runs of
2026-11-19 (pk2 / pk3 / pm3) and 2026-11-20 (pm4) on the SageMaker L4 box,
reduced and analysed on the laptop (`results/keyfix/keyfix_analysis.md`).
Design `docs/superpowers/specs/2026-11-18-prefix-codebook-design.md`, run
book `docs/keyfix_experiment.md`, proposal that led to pm4
`docs/keyfix_prefill_proposal.md`.

## 1. The question

Earlier cells on this branch had established two facts about 2-bit keys in
the Qwen3-TTS talker cache. Group scaling of the generated keys (g32, two
fp16 scalars per coordinate per 32 tokens) rescues 3-bit keys but not 2-bit
ones. Keeping the whole prefill (text plus reference audio, 42-67 keys) in
fp16 on top of g32 (g32pp) rescues 2-bit keys completely: 0/50 collapses,
CER 0.003 against fp16's 0.005. The catch is cost. On these short sentences
(87-759 keys) an fp16 prompt pools to 5.4 key bits per element at K2, so the
"2-bit" cache was not 2 bits.

The question for this experiment: can the prompt keys be quantized too, with
levels fitted on the prompt itself at prefill time, and still avoid
collapse? If yes, at how many bits, and does the shape of the levels matter?

## 2. What was built

Library (`turboquant/fitted_levels.py`, `MSECompressor(levels_mode=)`,
`TurboQuantV3(prefix_codebook, prefix_bits)`): two ways to fit
per-coordinate scalar levels on the prompt keys after the random rotation.

- `kmeans`: 2^b Lloyd-Max levels per coordinate fitted on the prompt's own
  values, stored as 2^b fp16 scalars per coordinate (L x 16 / P bits per
  prompt key element).
- `minmax`: one window per coordinate, the prompt's minimum and maximum,
  with 2^b uniform levels inside it, stored as 2 fp16 scalars per coordinate
  (32 / P bits per element). The endpoints are exact levels.

The Qwen cache (`turboquant_kv_cache.py`) compresses the prefix with that
codebook at prefill (`_compress_prefix`), the generated keys with the
ordinary g32pre path. Prompt values use the ordinary fixed Lloyd-Max value
codebook at the config's value bits in every cell. The recorder splices the
quantized prefix into the divergence run so attention metrics see the same
cache the audio saw. Cells, wav suffix `_pq=k<b>` / `_pq=m<b>`, CSV columns
`prefix_codebook` / `prefix_bits`, tools and tests followed.

## 3. Protocol

Same as every keyfix cell, so every comparison is paired per sentence.

- Model Qwen3-TTS talker (28 layers, GQA, mRoPE), clone mode, sampling
  T 0.9, seed 0.
- 50 LibriSpeech-PC sentences, reference audio from the same speaker.
- Configs fp16, K2V2@0 (keys 2 bit, values 2 bit, residual window 0) for
  all four cells; K3V3@0 for pm4 as well.
- Every codebook cell runs `--key-group 32 --pre-rope-keys --protect-prefix
  -1 --prefix-codebook <kmeans|minmax> --prefix-bits <b>`, so the only
  difference from g32pre is the prompt key quantizer, and the only
  difference from g32pp is that the prompt is quantized at all.
- Audio metrics: Whisper large-v3 CER and WER against the text, WavLM
  speaker similarity to the reference, collapse = CER > 0.5. Divergence
  metrics per layer and step on the talker rows (pos >= 33): attention JS
  divergence and top-1 agreement against the fp16 cache, output cosine, key
  and value cosine, relative key MSE.
- Reduction on the box (`tools/reduce_keyfix_div.py`), analysis on the
  laptop (`tools/analyze_keyfix.py`, sections 1-10).

GPU time: about 2.5 h per cell per config on the L4 (170-200 s per sentence).

## 4. Results

### 4.1 Audio, K2V2@0, 50 sentences

| cell | prompt keys | collapse | CER | WER | spk_sim | key bits pooled (at 1000 keys) |
|---|---|---|---|---|---|---|
| g32pre | fixed 2-bit codebook | 100 % | 1.04 | 1.26 | 0.75 | 3.00 (3.00) |
| pk2 | k-means 2 bit | 100 % | 0.99 | 1.06 | 0.68 | 3.03 (3.01) |
| pk3 | k-means 3 bit | 100 % | 0.97 | 1.02 | 0.68 | 3.43 (3.13) |
| pm3 | min-max 3 bit | 36 % | 0.33 | 0.44 | 0.92 | 3.11 (3.03) |
| pm4 | min-max 4 bit | 2 % | 0.034 | 0.057 | 0.954 | 3.30 (3.09) |
| g32pp | fp16 | 0 % | 0.003 | 0.010 | 0.992 | 5.44 (3.73) |
| fp16 | | 0 % | 0.005 | 0.014 | | |

Value bits are 2 in every row except g32pp (fp16 prompt values, 4.63
pooled). Bits exclude the per-vector norm, as in every cell.

### 4.2 Audio, K3V3@0

| cell | collapse | CER | WER | spk_sim | key bits pooled (at 1000 keys) |
|---|---|---|---|---|---|
| g32pre | 6 % | 0.13 | 0.18 | 0.957 | 4.00 (4.00) |
| pm4 | 0 % | 0.007 | 0.013 | 0.959 | 4.11 (4.03) |
| g32pp | 0 % | 0.004 | 0.013 | 0.992 | 6.25 (4.67) |

### 4.3 Paired tests (section 2 of the analysis)

| config | pair | rescued | lost | p |
|---|---|---|---|---|
| K2V2 | g32pre -> pm3 | 32 | 0 | < 0.001 |
| K2V2 | g32pre -> pm4 | 49 | 0 | < 0.001 |
| K2V2 | pm4 vs g32pp | 1 discordant | | 1.0 |
| K3V3 | g32pre -> pm4 | 3 | 0 | 0.25 |
| K3V3 | pm4 vs g32pp | 0 discordant | | |

pm4 and g32pp are statistically indistinguishable on collapse at both
configs. The K3V3 gain over g32pre (3 sentences) is real in direction but
not significant at n = 50.

### 4.4 Divergence, talker rows, K2V2@0

| cell | attn_js | attn_top1 | out_cos | cos_k | relmse_k | layer-5 attn_js |
|---|---|---|---|---|---|---|
| g32pre | 0.061 | 0.66 | 0.887 | 0.9918 | 0.016 | 0.226 |
| pk2 | 0.065 | 0.60 | 0.896 | 0.9938 | 0.011 | 0.219 |
| pk3 | 0.033 | 0.70 | 0.925 | 0.9969 | 0.005 | 0.122 |
| pm3 | 0.017 | 0.79 | 0.942 | 0.9965 | 0.006 | 0.055 |
| pm4 | 0.008 | 0.87 | 0.953 | 0.9979 | 0.004 | 0.028 |
| g32pp | 0.019 | 0.89 | 0.970 | 0.9921 | 0.015 | 0.036 |

cos_v is 0.9406 in every row except g32pp (0.9409): the prompt values are
identical across the codebook cells, as designed.

### 4.5 The one pm4 failure

Sentence 4 (fp16 5.4 s). Under pm4 K2V2 it runs to the 31.3 s generation
cap with CER 1.31 and speaker similarity 0.93: a runaway, the voice holds
while the content drifts. The same sentence runs to the cap under 16 of the
other 20 quantized settings in the summary, including pm3 (CER 0.45, just
under the threshold), g32 at K3V3 and pp at K3V3. Only g32pp (both configs),
g32pre at K3V3 and pm4 at K3V3 (CER 0.06) keep it intact. Section 9: every
collapse in pk2, pk3, pm3 and pm4 is a runaway, none an early stop. Section
10 cannot rank a single collapse (AUC 0.78, p 0.48).

Wavs for listening: `results/keyfix/wavs/keyfix_pm4/` (fp16, K3V3 pm4, K2V2
pm4 for sentence 4).

## 5. Reading

1. **Prompt keys can be quantized.** pm4 at K3V3 is a fully quantized cache
   at fp16 intelligibility: 0/50 collapses, CER 0.007, 4.11 key bits and 3
   value bits per element pooled, against g32pp's 6.25 / 5.44. pm4 at K2V2
   is the aggressive point: 1/50 at 3.30 key bits and 2 value bits. These
   are the two operating points for the thesis.
2. **The prompt keys were the carrier.** pm3 to pm4 changes only the prompt
   key bits (3 to 4); prompt values stay at 2 bits; collapse goes 36 % to
   2 %. The design's open confound, that the residual collapses might be the
   2-bit prompt values, is closed for collapse.
3. **Range, not shape.** pk3 has the lowest key reconstruction error of any
   cell (relmse 0.005) and collapses every sentence; pm3 at the same 3 bits
   with a worse error (0.006) collapses a third. k-means puts the levels at
   the quantile centres and shrinks the coordinate extremes; min-max keeps
   each coordinate's minimum and maximum as exact levels. Attention logits
   are inner products dominated by the few large coordinates of the few
   heavily attended prompt keys, so the extremes, not the average error,
   decide. MSE on the prompt keys is the wrong objective; more k-means
   variants are not worth running.
4. **The fourth bit matters because it halves the step inside an exact
   range.** With the endpoints fixed, 3 bits leaves 7 intervals across a
   coordinate's full prompt range, 4 bits leaves 15. The cost is 0.19 pooled
   key bits over pm3 and 0.30 over g32pre.
5. **Divergence finally agrees with the audio.** Across the earlier cells
   mean attention JS did not predict collapse (pp at K3V3 had lower JS than
   plain and collapsed 58 %). pm4 has the lowest talker attn_js of any
   quantized cell at both configs (0.008 at K2, below g32pp's 0.019; 0.004
   at K3) and the layer-5 peak, the g32-resistant layer, falls to 0.028.
   Along the sequence (section 7, K3V3) pm4's JS falls with decode position
   at every layer, the opposite of plain's drift signature.
6. **Speaker similarity is the open cost.** pm4 sits at 0.954 / 0.959,
   g32pp at 0.992. At K3V3 the remaining difference between them is prompt
   values at 3 bits against fp16. Keys carry intelligibility, values appear
   to carry timbre. Untested.

## 6. Limits

- One seed, 50 sentences, one speaker set. The K3V3 difference between
  g32pre and pm4 (3 sentences) is not significant on its own.
- Short sentences. The prompt is 42-67 of 87-759 keys, so the pooled bits
  overstate the prompt's share for long generation; the "at 1000 keys"
  column is the fairer long-form figure (pm4 K2 3.09, K3 4.03).
- Bit accounting excludes the per-vector norm everywhere; `memory_report()`
  includes it and reports slightly higher numbers.
- Collapse is a threshold on CER; sentence 4 under pm3 (CER 0.45, 31 s)
  shows a runaway can sit under it. Duration against fp16 is the more
  honest collapse signal and section 9 uses it.
- The cache writes reconstructed tensors back into the buffer, so these are
  fidelity results for the stored keys, not throughput or memory results
  for a packed kernel.

## 7. What is left

- Optional, zero code, ~1.5 GPU h: `CELLS="pm4" CFGS="fp16,K3V4@0" MAXPG=50`
  to price the speaker-similarity gap. Only if the thesis wants to say what
  the 0.04 costs in value bits.
- Seeds 1-2 for pm4 at K3V3 if the thesis leans on the 6 % to 0 % step over
  g32pre.
- Cells B (fp16 first 4 tokens) and D (dense-and-sparse prompt keys) from
  `docs/keyfix_prefill_proposal.md` are no longer needed for collapse.
- Thesis: the headline is "collapse 0/50 at 4.11 key bits (pm4, K3V3) and
  1/50 at 3.30 (pm4, K2V2), against 0/50 at 6.25 / 5.44 for the fp16
  prompt", with the mechanism claim that the prompt keys' per-coordinate
  range, not their MSE, is what the cache must preserve.
