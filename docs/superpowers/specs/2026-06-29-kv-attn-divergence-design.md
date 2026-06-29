# KV + attention-map divergence experiment (K4/V4, rw=24 vs rw=0)

## Goal

Quantify how K4/V4 KV-compression perturbs the talker's **attention maps** and **KV
vectors** at residual_window 24 vs 0, and produce the compressed audio — **without**
any ASR/CER (Whisper is run separately on the saved wavs). Pairs the static
rate-distortion view with the generation audio so the thesis can show the
*mismatch*: near-lossless reconstruction, yet generation collapses (= AR error
amplification). No baseline trial needed — the fp16 pass is only a substrate.

## Phases

**Phase 1 — counterfactual (this spec).** Deterministic, no model-file edits.
**Phase 2 — on-path dual-cache (follow-up).** Measures divergence on the actual
compressed trajectory that produced the audio; needs `turboquant_kv_cache.py`
surgery + an fp16 shadow cache. Built after Phase 1 results look right.

## Phase 1 design

New script `models/Qwen3-TTS/benchmarks/kv_attn_divergence_experiment.py`
(reuses benchmark helpers; the simpler `kv_recon_experiment.py` stays intact).

Per sentence:
1. **Produce audio** — `run_generation` under K4/V4 at rw=24 and rw=0 (sampling,
   seeded), save wavs to `outputs/` (recorder OFF).
2. **Measure divergence** — one fp16 `run_generation` with the recorder ON.

### Attention capture (counterfactual)

- Force the talker to eager attention (`config._attn_implementation = "eager"`), so
  `eager_attention_forward` (returns `attn_weights`) is the active path.
- **Monkeypatch** `modeling_qwen3_tts.eager_attention_forward` with a wrapper that
  calls the original to get `(attn_output, attn_fp16)`, then — when the recorder is
  active — for each rw: compress the fp16 `key`/`value` with `TurboQuantV3`,
  reconstruct, `repeat_kv`, recompute `attn_comp = softmax(scaling·q·rkᵀ + mask)`
  exactly like eager, and log `JS(attn_fp16, attn_comp)` plus KV cos/relMSE.
- Patch is installed only around the fp16 pass and removed after (no global state
  leak). It reads `module.layer_idx / num_key_value_groups / scaling`.

### Metrics

- **Attention:** Jensen-Shannon divergence (base-2, bounded [0,1]) over the key
  distribution, averaged over batch×heads×query-positions.
- **KV:** per-vector cosine + relative MSE (reuse `_kv_recon_errors`).

### Cost control

- `--step-stride` (record every k-th decode position; recomputing the growing key
  each step is O(seq²)), `--max-per-group`, `--layer-stride` (optional). Long-audio
  stays tractable; any subsampling is logged, not silent.

### Output

- `results/kv_attn_k4v4_rw24_vs_rw0.csv`:
  `group, idx, layer, pos, rw, attn_js, cos_k, cos_v, relmse_k, relmse_v`.
- Console summary: mean-by-rw for attn_js + KV, with a `diff(0−24)` row.
- Audio: `outputs/qwen_<group>_<idx>_..._K4_V4_rw=<rw>.wav`.

## Caveat (stated in the report)

Phase-1 divergence is on the **fp16 trajectory**; the saved audio is the
**compressed** trajectory — consistent but not the same path. Phase 2 closes this.

## Verification

- GPU-free unit tests: `js_divergence` (identical→0, disjoint→1, symmetric);
  recorder logs one row per (layer, rw) on a synthetic eager call; `summarize`
  diff-row math; wav-name/path reuse.
- Box smoke: 2–3 sentences → CSV has both rws + audio saved; sanity
  `attn_js(rw=0) > attn_js(rw=24)` (quantizing all keys moves attention more).

## Critical files

- `models/Qwen3-TTS/benchmarks/kv_attn_divergence_experiment.py` (new)
- `tests/test_kv_attn_divergence.py` (new)
- reuses `benchmark_qwen3tts_real` (`run_generation`, `_extract_fp16_kv`,
  `_kv_recon_errors`, `_resolve_voice`), `turboquant` (`TurboQuantV3`,
  `decode_overrides`, `set_global_seed`, `iter_eval_items`), and
  `modeling_qwen3_tts` (`eager_attention_forward`, `repeat_kv`).
