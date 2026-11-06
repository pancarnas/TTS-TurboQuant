# Follow-up experiments after OmniKVQuant (arXiv:2609.11582)

Status: proposal, nothing implemented. Pick the options below, then the code
goes in behind config flags with CPU unit tests; GPU runs go out as a checklist.

## Why

OmniKVQuant (Yoo et al., Sep 2026) applies MSE-only TurboQuant to Qwen2.5-Omni
and Qwen3-Omni and sees the same pattern we see: K2V2 is nearly free on one Qwen
model and collapses on another. They fix it with (a) per-group min-max scaling of
rotated keys over 32 consecutive tokens and (b) a calibrated value rotation per
modality. They do not explain the collapse, do not control for RoPE, and do not
test autoregressive decode beyond captioning.

Our open question is sharper than theirs because we have two models that differ
in exactly the right ways:

| | Qwen3-TTS (collapses at K2) | VALL-E-X (no collapse) |
|---|---|---|
| positions | multimodal RoPE on keys | additive sinusoidal, keys carry no position |
| attention | GQA | MHA |
| text/audio layout | text prefix + codec tokens | phoneme prefix + codec tokens |

Three candidate causes of the collapse, and the experiment that separates them:

| cause | prediction | experiment |
|---|---|---|
| key range drift (paper's claim) | group scaling rescues K2 on Qwen | E1 |
| RoPE spreads key range by position | pre-RoPE quantization rescues K2 on Qwen, group scaling helps less | E2 |
| structural (GQA, layer-5 attention response) | neither rescues; attn_js hotspot survives | E1 + E2 both negative |

Outcome either way is a thesis result: a mechanism claim backed by a control the
published paper lacks.

## E1. Per-group key scaling (their key fix)

Cost: ~1 day code, one Qwen divergence run + one VALL-E-X grid (~6-10 GPU hours).

Code changes

- `turboquant/config.py`: add `key_group: int = 0` (0 = current behaviour).
- `turboquant/compressors_v3.py` `MSECompressor.compress/decompress`: when
  `key_group > 0`, after rotation reshape to `(B, H, S/g, g, D)`, take per-group
  per-coordinate min and max, map to [0, 1], quantize on a uniform 2^bits grid
  (not the Lloyd-Max centroids), store `min` and `scale` as fp16 next to the
  norms. Decompress reverses it. `memory_bytes` counts the metadata honestly:
  two fp16 scalars per coordinate per g tokens = 32/g bits per element (1.0 bit
  at g = 32). Report that number; do not repeat the paper's undercount.
- Both caches (`models/VALL-E-X/turboquant_cache.py`,
  `models/Qwen3-TTS/.../turboquant_kv_cache.py`) `_legacy_compress_overflow`:
  round the compressed span end down to a multiple of `key_group` so groups are
  whole; the partial tail stays fp16 until the block fills (same as the paper).
- `tools/score_wav_dir.py` / wav naming: add a `_g32` suffix so configs do not
  collide with existing outputs.

Tests (CPU, model-free)

- round-trip: group-scaled K2 on a narrow-band synthetic key window has lower
  MSE than fixed-codebook K2; identical output when `key_group = 0`.
- cache: compressed span boundaries are multiples of g; tail tokens bit-exact.
- memory report: metadata bytes present and equal to the formula.

Runs

1. `13_qwen_divergence.sh` at K2V2 rw0 and K2V4 rw0, `key_group` 0 vs 32,
   seed 0. Read per-layer `attn_js`, `cos_k`, CER. Decision metric: does the
   layer-5 `attn_js` hotspot go away, and does per-sentence collapse rate drop?
2. If (1) is positive, VALL-E-X grid at the same cells to confirm it does not
   hurt the model that was already fine.

## E2. Pre-RoPE key quantization (the control the paper lacks)

Cost: ~1 day code, Qwen only, one divergence run (~3-5 GPU hours).

Code changes

- `turboquant/config.py`: add `pre_rope_keys: bool = False`.
- Qwen cache: the attention layer already passes `cos`, `sin`, `cache_position`
  in `cache_kwargs` each step. Keep a per-position buffer of cos/sin. In
  `_legacy_compress_overflow`, when the flag is set, undo RoPE on the aged-out
  span (same rotary function with negated sin), quantize-dequantize, re-apply
  RoPE, write back. Attention still reads post-RoPE keys; nothing else changes.
- VALL-E-X: not applicable (no RoPE). This is the point: its keys are already
  "pre-RoPE".

Tests (CPU)

- undo-then-redo RoPE on random keys is identity to fp16 tolerance.
- with `pre_rope_keys=True` and 8-bit keys, output matches `False` closely
  (quantization error negligible, so RoPE handling must be exact).

Runs

- Same two cells as E1 run 1, `pre_rope_keys` True vs False, `key_group = 0`.
- Optional 2x2: pre-RoPE x group scaling, to see if the effects stack.

## E3. Modality-aware value rotation (their value fix)

Cost: 2-3 days code, calibration pass per model, one grid each.
Recommendation: only if E1 leaves a gap at K2. Their own ablation says this is
the smaller effect and it lowered multiple-choice accuracy on the MoE model.

Code changes

- `tools/calibrate_value_rotation.py`: run 20-50 sentences, hook attention to
  collect per-layer, per-head attention mass per token and the value vectors,
  split by modality (text prefix vs codec tokens; both models expose the
  boundary), accumulate attention-weighted covariance, eigendecompose, save
  `U_m` per (layer, head, modality) to a `.pt`.
- `MSECompressor`: optional per-modality rotation `R_m = Pi @ U_m^T` selected
  by a per-token modality id passed through `compress_kv`.
- Caches: track which positions are text vs audio (known at prefill).

Runs

- VALL-E-X and Qwen grids at V2 with K held at 4, rotation shared vs
  per-modality. Metric: `cos_v`, `out_cos`, CER, SpkSim.

## E4. Cheap diagnostics that need no new quantizer (half a day)

Can run before any of the above and may already answer the question.

- Key range per 32-token window on Qwen3-TTS vs VALL-E-X from the existing
  divergence recorder (their Table 6 analogue). If VALL-E-X windows are as
  narrow as Qwen's, range drift cannot be the discriminating cause.
- Same statistic on Qwen with RoPE undone offline (keys are logged; cos/sin
  are recomputable from positions). Tells us before E2 whether RoPE is
  widening the range.
- GQA check: per-head `attn_js` at layer 5, grouped by KV head. If the damage
  concentrates in KV heads shared by the most query heads, that points at
  structure rather than range.

## Decision table

| option | answers | GPU hours | code days | pick |
|---|---|---|---|---|
| E4 diagnostics | is drift even different between the models | 0-1 | 0.5 | [ ] |
| E1 group key scaling | does the paper's fix transfer to TTS | 6-10 | 1 | [ ] |
| E2 pre-RoPE | is it RoPE | 3-5 | 1 | [ ] |
| E3 value rotation | does the value fix matter for TTS | 10-15 | 2-3 | [ ] |

Suggested order: E4 first (free, sharpens the hypothesis), then E1 and E2
together on the Qwen divergence arm (they share the run script), E3 only if
needed.

## Thesis placement

- Related work: cite OmniKVQuant as the independent confirmation that MSE-only
  TurboQuant is the deployed variant and that K2 is model-dependent on the Qwen
  family; note they offer no mechanism.
- Results: E1/E2 become a short "mechanism control" subsection. One figure:
  layer-wise `attn_js` for baseline K2, +group scaling, +pre-RoPE.
- Limitations: metadata overhead of group scaling (1 bit/elem at fp16, ~0.27 at
  MXFP4-style 4-bit metadata); state our own nibble-granularity bit accounting
  for 3-bit so the same critique cannot be raised against us.
