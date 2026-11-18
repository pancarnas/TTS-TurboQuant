# Prefix codebook: quantized prompt keys with fitted levels + g32pre generation

Date 2026-11-18. Branch `exp/keyfix-group-prerope`. Run book `docs/keyfix_experiment.md`.

## Intent

A fully quantized Qwen3-TTS KV cache at 2-bit generated keys with no fp16
prefix. The `pp` cells showed that the 2-bit collapse lives in the 53 prompt
keys (prefill: reference transcript + reference-audio codec frames), which
are stationary, twice as spread as generated keys, and carry 23-73 % of the
attention mass. Group scaling (g32, with pre-RoPE) already repairs the
generated keys. This experiment gives the prompt its own quantizer: levels
fitted on the prompt at prefill time (per-coordinate 1-D k-means, or a single
min-max window), and keeps g32pre on everything generated. The thesis claim
it enables: a pure-quantization rescue of 2-bit keys, with the prompt cost
accounted honestly.

## Goals

- Cells `pk2`, `pk3` (k-means prompt levels at 2 / 3 bits) and `pm3` (one
  min-max window over the prompt at 3 bits, the matched-cost comparator),
  each with `--key-group 32 --pre-rope-keys` on the generated keys, configs
  `fp16,K2V2@0`, 50 LibriSpeech-PC sentences, seed 0, same protocol as the
  existing cells. Success = collapse rate and CER/spk_sim comparable to
  `g32pp` (0/50, CER 0.003) at a pooled key cost near 3.2 bits per element.
- Every existing cell (`plain`, `g32`, `pre`, `g32pre`, `pp`, `g32pp`) stays
  byte-for-byte reproducible: same flags, same wav names, same CSV columns
  with the same values, same config labels, same defaults.

## Non-goals

- Prompt values: they stay on the fixed Lloyd-Max codebook at `value_bits`
  (keygauss: prompt values are near-Gaussian, spread 0.85, offset 0.15).
- A keys-only / values-only variant of `pp`, the prefix sweep (`ppN`), and
  MXFP4 metadata: separate experiments.
- VALL-E-X: its cache ignores `protect_prefix` and ignores this too.
- Splitting the prompt into text and reference-audio streams.

## Anti-goals

- Any fp16 token in the compressed region (that is what `pp` already is).
- A second code path for the generated keys. g32pre stays untouched.
- Silent changes to old cells. If a default moves, the experiment is invalid.

## Constraints

- `turboquant/` is the shared library (ask before editing was the rule; this
  design is the ask). Changes there are additive and default-off.
- Divergence rows, wav names, score labels, `--resume` and the smoke
  validator must all carry the new setting so cells cannot be mixed up.
- Box: L4, torch 2.6.0+cu124, chat client mangles quotes; cells are one-word
  `CELLS` entries. The prefill is 44-67 keys, so the k-means fit is tiny.
- Functions <= 50 lines, files <= 300 lines where new; existing oversized files
  get the smallest diff.

## Approaches considered

- **A. Prefix compressor beside the existing one (chosen).** The cache already
  knows the prefix span from `protect_prefix`; instead of leaving it fp16 it
  compresses that span once at prefill with fitted-levels keys. Generated
  keys keep the g32pre path. Recorder, names, labels, smoke and run script
  are extended, not duplicated. One implementation of the fitted levels
  serves the compressor and the keygauss diagnostic.
- B. Generalize the group path (prefix = one big group with fitted levels).
  One compressor would hold two bit widths, which breaks index packing, and
  the group metadata shape (fp16 min/scale) is wrong for 2^bits levels.
- C. Offline codebook fitted on other sentences. Static-affine results say
  the prompt shape is partly per-sentence; dropped.

## Design

### Config (`turboquant/config.py`)

Two additive fields, both default off:

- `prefix_codebook: str = ""` — `""` (prefix stays fp16, today's `pp`),
  `"kmeans"` (per-coordinate 1-D k-means levels fitted on the prefix),
  `"minmax"` (one min-max window over the prefix, uniform levels).
- `prefix_bits: int = 0` — bits for the prefix keys. `0` = follow each
  config's `key_bits` (so a K8V8 config in the same run gets an 8-bit
  prompt and a K2V2 config a 2-bit one); `N > 0` = literally N for every
  config in the run. The thesis cells use literal values (`pk3` = 3-bit
  prompt over 2-bit generated keys); the smoke uses `0` so its K8V8 config
  doubles as a losslessness check.

Rules: `prefix_codebook` requires `protect_prefix == -1` (the whole prefill);
any other `protect_prefix` with a codebook is a config error. `protect_prefix`
keeps its meaning as the span definition; the codebook field decides fp16 vs
quantized. Prefix values are compressed with the normal value compressor.
The effective prefix bits of a config are `prefix_bits or key_bits`, capped
at 8 like every other bit width.

### Fitted levels (`turboquant/fitted_levels.py`, new, small)

Pure tensor functions, CPU-testable, used by both the compressor and
`keygauss_core` (which today has its own `quant_kmeans`; it switches to this
module so there is one implementation):

- `fit_levels(z, bits, mode, iters=20)` — `z` is `(N, S, D)` rotated unit
  coordinates (N = B*H blocks, S tokens, D coords). Returns `levels (N, L, D)`
  fp32. `kmeans`: init at the S-sample quantiles, Lloyd iterations (assign
  nearest, move to member mean, empty clusters keep their level), fixed
  iteration count so results are deterministic. `minmax`: `L` evenly spaced
  levels from the per-coordinate min to max.
- `assign(z, levels)` -> indices `(N, S, D)` int32; `reconstruct(indices,
  levels)` -> `(N, S, D)`.
- Levels are rounded to fp16 before assignment so compress and decompress
  share the exact grid (same rule as the group path).
- Stored metadata differs by mode, and so does its cost: `kmeans` stores the
  `L` fp16 levels per coordinate; `minmax` stores only the two fp16 endpoints
  per coordinate (the levels are implied, exactly as the group path stores
  min/scale). Metadata is therefore `L` scalars per coordinate per block for
  kmeans and `2` for minmax. This is what makes `pm3` the matched-cost
  comparator of `pk2` (at P = 53: 3.6 vs 3.2 bits per prompt key element).

### Compressor (`turboquant/compressors_v3.py`)

- `MSECompressor` gains `levels_mode: str = ""`. When set, the codebook stage
  is `fit_levels` + `assign` over the whole block passed to `compress`
  (the prefix), with `levels` stored fp16 in the output dict; `decompress`
  uses `reconstruct`. Packing of indices and norms is unchanged.
- `compressed_bytes_per_token(head_dim, bits, group=0, meta_scalars=0,
  block=0)`: trailing keyword arguments (VALL-E-X calls it positionally);
  when `meta_scalars > 0`, adds `meta_scalars * head_dim * 2 / block` bytes,
  the fp16 metadata amortized over the block's tokens (`meta_scalars` = `L`
  for kmeans, `2` for minmax). This is the single accounting formula; the
  cache's `memory_report` and `memory_bytes` call it.
- `TurboQuantV3(..., prefix_codebook="", prefix_bits=0)` builds a
  `prefix_key_compressor` (same rotation seed as the key compressor, own
  bits, `levels_mode`) when a codebook is set. `compress_kv(keys, values,
  prefix=0)`: with `prefix > 0`, `[0, prefix)` keys go through the prefix
  compressor and `[0, prefix)` values through the value compressor; the rest
  is handled exactly as today (residual window, group alignment starts at
  `prefix`). Output dicts carry a `"prefix"` entry; `decompress_kv`
  concatenates `[prefix | compressed | fp16 tail]`. `prefix=0` (default)
  is the current behaviour, bit for bit.

### Qwen cache (`models/Qwen3-TTS/qwen_tts/core/models/turboquant_kv_cache.py`)

- On the prefill update (`s == 0`) with a codebook set: compress `[0, pfx)`
  immediately through `compress_kv(..., prefix=pfx)` on that span, write the
  lossy reconstruction back into the buffer (same write-back pattern as
  `_legacy_compress_overflow`), append a `{"blob", "span"}` chunk, and leave
  `_prefix_len` / `_compressed_upto` as they are so the generated-key path
  (group alignment at the prefix end, residual window) is untouched.
- Pre-RoPE: when `pre_rope_keys` is on, the prefix keys are un-RoPEd before
  compression and re-RoPEd after, with the same helpers as the overflow path.
- `memory_report`: with a codebook, prefix tokens are counted at the prefix
  compressor's bytes (keys) and the value compressor's bytes (values), not
  fp16. Without a codebook, unchanged.
- `track_only=True` keeps the buffer pristine (analytic accounting only), as
  for the other paths.

### Recorder (`models/Qwen3-TTS/benchmarks/kv_attn_divergence_experiment.py`)

- `DivergenceRecorder(..., prefix_codebook="", prefix_bits=0)`; `_comp`
  passes them to `TurboQuantV3`. In `record`, with a codebook set, the
  generated keys are compressed exactly as today (full `[0, split_at)`
  tensor, groups aligned at position 0, which is how the `g32pp` rows were
  recorded), and the prompt is spliced back **quantized through the prefix
  compressor** instead of fp16. The rows of `pk*` cells then differ from
  `g32pp` rows only in the prompt, so the attention comparison between them
  is clean; the cache (and therefore the wavs) aligns groups at the prefix
  end in both, as it does today. The KV reconstruction metrics cover
  `[0, split_at)` (prompt included) instead of `[pfx, split_at)`, because
  the prompt is now lossy.
- Two new CSV columns appended after `protect_prefix` in `COLUMNS`:
  `prefix_codebook` (string: `""`, `kmeans`, `minmax`) and `prefix_bits`
  (the run-wide CLI value, 0 or N). `load_done` filters on both when present
  (old CSVs without the columns hold no codebook rows).
- CLI: `--prefix-codebook {kmeans,minmax}` and `--prefix-bits N`; printed in
  the key-fix banner. `make_config` carries both.
- Wav naming: with a codebook, the segment is `_pq=k2` / `_pq=m3` (k =
  kmeans, m = minmax, digit = the config's **effective** prefix bits, so a
  K8V8 config under `--prefix-bits 0` writes `_pq=k8`) **instead of**
  `_pp=prompt`; fp16 wavs keep no segment. Old names are untouched.

### Scoring and tools

- `tools/score_wav_dir.py` and `tools/check_audio.py`: regex gains
  `(?:_pq=(?P<pq>[km]\d+))?`; label suffix gains `+pk2` / `+pm3`. Existing
  labels unchanged.
- `tools/summarize_keyfix.py`, `tools/analyze_keyfix.py`,
  `tools/reduce_keyfix_div.py`: `prefix_codebook` joins the config columns;
  cells `pk2`, `pk3`, `pm3` join the cell lists with labels and styles;
  effective-bit accounting for a codebook cell uses the prefix formula
  (`prefix_bits + meta_scalars*16/P` per prompt key element, `meta_scalars`
  = L for kmeans, 2 for minmax) and `key_bits + 1` (g32) for generated keys.
  The per-vector fp16 norm stays excluded everywhere, as section 8 already
  does, so the columns remain comparable.
- `tools/validate_keyfix_smoke.py`: cell `pk` (`--key-group 32
  --pre-rope-keys --protect-prefix -1 --prefix-codebook kmeans
  --prefix-bits 0`, so the prompt follows each config's key bits): columns
  present with the right values; K2V2 prefill rows are lossy (`cos_k < 1`)
  with `relmse_k` below plain's K2V2 prefill rows; K8V8 prefill rows are
  near-exact (256 levels on <= 67 tokens reproduce the data: `cos_k >
  0.999`); K2V2 wav differs from plain and from pp.

### Scripts

- `scripts/gpu/03_smoke_keyfix.sh`: cell `pk` added to `ALL_CELLS` and the
  case table.
- `scripts/gpu/14_qwen_keyfix.sh`: cells `pk2`, `pk3`, `pm3` in `cell_flags`
  (each `--key-group 32 --pre-rope-keys --protect-prefix -1
  --prefix-codebook <mode> --prefix-bits <b>`); default `CELLS` unchanged.
  Run: `CELLS="pk2 pk3 pm3" CFGS="fp16,K2V2@0" MAXPG=50`.

### Tests (CPU, `tests/`)

- `test_fitted_levels.py`: minmax endpoints exact; kmeans deterministic and
  never worse than minmax on the same block; `L >= S` is lossless; fp16 grid
  agreement between assign and reconstruct.
- `test_prefix_codebook.py`: compressor round trip with `prefix > 0`
  (prefix lossy, tail as before, `prefix=0` identical to old output);
  bytes-per-token with levels; cache prefill compression (prefix lossy,
  generated keys unchanged vs a `pp` cache fed the same tokens, pre-RoPE
  on/off); `memory_report` counts the prefix at codebook bytes; recorder
  prefill rows lossy and metrics cover the prompt; naming, parse, label,
  resume filter; config validation (codebook without `protect_prefix=-1`
  rejected); defaults off.
- Existing tests (`test_keyfix.py`, `test_protect_prefix.py`, cache readback,
  scorer, check_audio) must pass unchanged: they are the reproducibility
  guard for the old cells.

## Commit order

The uncommitted keygauss diagnostic (`keygauss_core.py` per-segment errors,
`quant_kmeans`, `test_keygauss_segments.py`) lands first as its own commit,
so the box can run `BITS=2,3,4 bash scripts/gpu/15_qwen_keygauss.sh` with
the per-segment table while this is built. The `fitted_levels.py` refactor
of `quant_kmeans` is a later, behaviour-preserving commit (the fp16-rounded
grid shifts its numbers slightly; the test thresholds allow it).

## Left to the builder

- Exact vectorization of `fit_levels` (one-hot vs scatter) and the Lloyd
  iteration count (20 is the diagnostic's value).
- Where the `"prefix"` dict lives inside the compressed dict and how the
  cache's chunk list labels it.
- Plot colours/markers for the three new cells in `analyze_keyfix.py`.
- Whether `keygauss_core.quant_kmeans` becomes a thin wrapper or is deleted
  in favour of the new module (tests in `test_keygauss_segments.py` must
  keep passing either way).
