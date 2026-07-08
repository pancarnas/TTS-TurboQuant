# KV-cache quantization: Qwen3-TTS vs VALL-E-X — findings

TurboQuant KV-cache compression on two autoregressive TTS models, same eval
sentences, same quantizer, same scorer. Headline: **the catastrophic collapse
TurboQuant causes on Qwen3-TTS does not occur on VALL-E-X** — and the reason is
architectural (how attention responds to key error), not reconstruction
fidelity.

Data: `results/combined_summary.csv` (per model × dataset × config: CER, WER,
collapse%, spkSim, cos_k, cos_v). Sources: Qwen `wav_scores.csv` +
`kv_attn_100pg_noprotect.csv` (pl=0); VALL-E `vallex_wav_scores.csv` +
`vallex_attn_divergence.csv` (pl=2). Collapse% = fraction of clips with
CER > 0.5. cos_k/cos_v averaged over unprotected layers only.

## 1. Robustness — collapse% on librispeech_pc

Seconds-matched residual windows: Qwen rw24 ≈ 1.9 s ↔ VALL-E rw128 ≈ 1.7 s (both
75 Hz? no — Qwen 12.5 Hz, VALL-E 75 Hz; windows chosen to match wall-clock).

| config (key/val bits) | Qwen collapse% | VALL-E collapse% |
|---|---|---|
| fp16 baseline | 0 | 7 |
| K4V4, seconds-matched window | 29 (@24) | 8 (@128) |
| K4V4, no window (@0) | 63 | 9 |
| K3V3, seconds-matched window | **100** (@24) | **11** (@128) |
| K3V3, short window | — | 18 (@64) |

Qwen: 3-bit keys = total collapse (100%, every dataset), and removing the fp16
window nearly doubles collapse (29→63%). VALL-E: every config stays within a few
points of its own fp16 baseline; 3-bit keys cost ~+4–11pp, not +100pp. The
residual window, decisive on Qwen, barely matters on VALL-E (@0 9% ≈ @128 8%).

## 2. Mechanism — reconstruction fidelity does NOT predict robustness

cos_k (key cosine similarity, quantized vs exact, unprotected layers,
librispeech_pc):

| | Qwen cos_k | VALL-E cos_k |
|---|---|---|
| K4 keys | 0.9987 | 0.9955 |
| K3 keys | 0.9954 | 0.9835 |

VALL-E's keys reconstruct **worse** at both bit-widths (64-dim heads vs Qwen's
128) — yet VALL-E is the robust one. So the vulnerability is not "how well the
keys survive quantization." It is how the attention softmax and the AR sampling
loop respond to a given key perturbation: Qwen's GQA (shared KV heads) +
peaky attention amplify a key error into a flipped attended position, which the
autoregressive loop then feeds into a runaway cascade; VALL-E's vanilla MHA with
flat, multimodal acoustic distributions absorbs the same (larger) perturbation
as a benign resample. This matches the divergence data (VALL-E attn_js ~4× lower
than Qwen at matched bits despite worse cos_k) and the teacher-forced PPL run
(ΔPPL ≈ 0 on VALL-E even at K3, KL small and bit-ordered, token flips frequent
but harmless).

## 3. Key bits ≫ value bits (both models)

Dropping value bits is nearly free; dropping key bits is where damage lives.
Qwen K4V3@24 ≈ K4V4@24 (collapse 29% both, librispeech); K3V3@24 = 100%.
VALL-E K4V3@0 (14%) ≈ K4V4@0 (9%); the jump is at 3-bit keys. cos_k confirms:
V4→V3 leaves cos_k unchanged, K4→K3 drops it sharply.

## 4. NAR quantization is free (VALL-E only)

The `-nar` arms (NAR stage quantized, AR untouched) are consistently the
**best** rows: highest spkSim (~0.99, vs ~0.96 for AR arms), lowest CER/WER,
often at or below fp16. Because AR runs at fp16 the token stream is identical to
baseline; only fine acoustic codebooks change, inaudibly — even at 3-bit. The
`-both` arms ≈ the plain AR arms: the two damage channels do not compound.
Practical: NAR-stage compute can be quantized aggressively for free.

## 5. Long-form (libritts_long) — the model's limit, not quantization's

At 20 s–160 s passages VALL-E-X collapses ~87–90% **at fp16** — it was not
trained for long-form and hits its own ceiling. Critically, quantization adds
essentially nothing on top: fp16 87% vs the worst quantized arm 90%, ΔWER within
noise, spkSim still ~0.98 (nar arms ~0.996). So even where the model itself
fails, KV quantization does not make it worse. The operating limit found here is
VALL-E-X's context length, not TurboQuant.

## Caveats

- **Qwen WER not yet computed** — Qwen wavs were scored before the WER column
  existed; the combined table's Qwen WER is blank. Re-score to fill:
  `python tools/score_wav_dir.py --audio-dir models/Qwen3-TTS/benchmarks/outputs
  --data-dir data --device cuda --out results/qwen_wav_scores_wer.csv`.
- **Protection levels differ**: Qwen run pl=0, VALL-E pl=2. The pl=0 VALL-E arm
  (`vallex_pl0_scores.csv`) confirms protection is not what saves VALL-E — K3V3
  at pl=0 still only mildly degrades vs its fp16.
- **Config grids differ**: Qwen ran rw ∈ {24, 0}, VALL-E rw ∈ {0, 64, 128};
  compare via the seconds-matched pairing above, not label-for-label.
- **Divergence recording differs**: Qwen during free generation, VALL-E
  teacher-forced; both measure quantized-vs-exact cos on the same prefix per
  step, comparable but not identical decode paths. VALL-E divergence covers
  librispeech_pc only (the group with ground-truth audio).
- **ellav_hard** is a reconstruction of the ELLA-V hard set, not the canonical
  100; its high fp16 CER reflects tongue-twister difficulty, not a pipeline bug.
