# Key-fix experiment: group key scaling (E1) and pre-RoPE keys (E2)

Branch `exp/keyfix-group-prerope`. Background and the full option list are in
`docs/followup_omnikvquant_plan.md`; this file is the run book.

## Question

Qwen3-TTS collapses at 2-bit keys, VALL-E-X does not. OmniKVQuant
(arXiv:2609.11582) fixes a similar collapse on Qwen-Omni with per-group min-max
key scaling and attributes it to "temporal key drift". They never control for
RoPE. Our two models differ in exactly that: Qwen3-TTS keys carry multimodal
RoPE, VALL-E-X keys carry no position at all.

| cell | flag | what changes |
|---|---|---|
| plain | (none) | fixed Lloyd-Max codebook, post-RoPE keys (today's baseline) |
| g32 | `--key-group 32` | rotated keys min-max scaled per 32 consecutive tokens, uniform 2^bits grid, fp16 min/scale stored (the paper's fix) |
| pre | `--pre-rope-keys` | RoPE undone on the span that ages out of the window, quantized, RoPE re-applied; attention still reads post-RoPE keys |
| g32pre | both | the two stacked |
| pp | `--protect-prefix -1` | the prefill (prompt: text + reference audio, 44-67 tokens) stays fp16, keys and values; everything generated is compressed as in plain |
| g32pp | `--key-group 32 --protect-prefix -1` | group scaling on the generated keys, prompt exact |
| pk2 | `--key-group 32 --pre-rope-keys --protect-prefix -1 --prefix-codebook kmeans --prefix-bits 2` | g32pre on the generated keys; prompt keys quantized at 2 bits with k-means levels fitted on the prompt. Prompt cost 3.2 bits per key element (pooled 3.04 on these sentences) |
| pk3 | same with `--prefix-bits 3` | 3-bit k-means prompt. Prompt cost 5.4 bits (pooled 3.42) |
| pm3 | `--key-group 32 --pre-rope-keys --protect-prefix -1 --prefix-codebook minmax --prefix-bits 3` | 3-bit prompt on one min-max window. Prompt cost 3.6 bits (pooled 3.11) |

`pp` follows the keygauss diagnostic (2026-11-17): 23-73% of the talker's
attention mass per layer sits on the prompt keys, so the hypothesis is that
most of the 2-bit collapse comes from the few dozen prompt keys, not the
hundreds of generated ones. Bit accounting is honest: with prompt length P,
sequence S and b bits, keys cost (16P + b(S-P))/S bits per element on
average; `memory_report()` counts the prefix at fp16. VALL-E-X's cache ignores
`protect_prefix` (not implemented there).

Protection cells (`pp`, `ppN`, keys-only/values-only) are diagnostics, not
candidates: they run only when a fully quantized cell fails and we need to know
which part of the prompt is the carrier. `g32pp` stays as the reference
ceiling.

Prefix codebook (`pk2`, `pk3`, `pm3`): the candidate that follows from `pp`.
The prompt keys are quantized at prefill with levels fitted on the prompt
itself (k-means: 2^b levels per coordinate, stored fp16; minmax: one window, two
endpoints); prompt values use the fixed codebook, generated keys stay g32pre.
Design in `docs/superpowers/specs/2026-11-18-prefix-codebook-design.md`.
`keygauss_core.quant_kmeans` now shares `turboquant/fitted_levels.py` and its
init quantiles changed from (i+1)/(L+1) to (i+0.5)/L, so k-means numbers from
before 2026-11-19 may differ slightly.

| if K2 is rescued by | conclusion |
|---|---|
| g32 (and pre does little) | range drift, the paper's story transfers to TTS |
| pre (and g32 does little) | RoPE spreads the key range; the paper's diagnosis is a positional artefact |
| neither; layer-5 `attn_js` hotspot survives | structural (GQA / attention response), our thesis mechanism stands |

Both flags are implemented in the shared `TurboQuantV3` / cache layer, so E1
runs unchanged on VALL-E-X (`benchmark_vallex_real.py --key-group 32`). E2 only
applies to Qwen3-TTS.

## Order of operations on the GPU box

```bash
git fetch origin && git checkout exp/keyfix-group-prerope
source .venv/bin/activate            # same env as the campaign (00_setup.sh)
# data/ must already hold librispeech_pc (01_fetch_data.sh)

# 1. smoke — MUST print "KEYFIX SMOKE PASS" (10-25 min)
bash scripts/gpu/03_smoke_keyfix.sh
# re-run ONE cell after a code fix (keeps the other cells' CSVs and wavs):
CELLS="pre" bash scripts/gpu/03_smoke_keyfix.sh
# the validator expects every cell from the same run: CELLS=pk alone against
# older smoke CSVs fails the column checks for the old cells, so re-run the
# default ALL_CELLS once after pulling this branch

# 2. full run, Qwen, seed 0, three cells (9-18 h at MAXPG=100; see knobs)
nohup bash scripts/gpu/14_qwen_keyfix.sh > logs/keyfix.log 2>&1 &

# 3. read results/keyfix_summary.md (re-runnable any time, also mid-run)
python tools/summarize_keyfix.py --results results \
    --plain-div results/qwen_div_seed0.csv --plain-scores results/qwen_seed_scores.csv
```

Prefix-codebook cells (after the smoke prints PASS; seed 0, 50 sentences):

```bash
CELLS="pk2 pk3 pm3" CFGS="fp16,K2V2@0" MAXPG=50 nohup bash scripts/gpu/14_qwen_keyfix.sh > logs/keyfix_pq.log 2>&1 &
```

Shorter first pass: `MAXPG=50 bash scripts/gpu/14_qwen_keyfix.sh` halves the
time; `CELLS="g32 pre"` drops the stacked cell. Everything resumes per sentence
(`--resume`), so a killed run is restarted with the same command.

The `plain` baseline for K2V2@0 and K3V3@0 is what `13_qwen_divergence.sh`
already produced at seed 0; the summary tool takes it via `--plain-div` /
`--plain-scores`. Add `plain` to `CELLS` only if you want it regenerated under
this run's config list (K2V4@0 is new).

## What the smoke checks

- unit tests (compressor, both caches, RoPE undo/redo, recorder, parsers)
- 2 sentences x {fp16, K8V8@0, K2V2@0} through the divergence experiment, once
  per cell
- fp16 wavs carry no flag segment (one baseline per sentence for the scorer)
- K2V2 wavs differ between cells (the flags reach generation, no silent no-op)
- K8V8 under `--pre-rope-keys` has cos_k > 0.999 and attn_js < 1e-3 on the real
  mRoPE tables: with near-lossless keys, any error left would be a wrong RoPE
  inverse
- CSV rows carry `key_group` / `pre_rope` / `protect_prefix` /
  `prefix_codebook` / `prefix_bits` with the cell's values
- the `pk` cell: its K2V2 prefill rows are lossy (cos_k < 1) with lower
  relmse_k than plain's, its K8V8 prefill rows have cos_k > 0.999, and its K2V2
  wav differs from plain and from pp

It prints the K2V2 cos_k / attn_js of the three cells as information only. The
smoke never asserts the scientific outcome.

## Resource estimate

| | value | note |
|---|---|---|
| GPU memory | 16 GB works, 24 GB comfortable | identical to `13_qwen_divergence.sh`; the new state is tiny (cos/sin buffer under 20 MB, group metadata 1 bit per key element) |
| transient peak | +0.6 GB during the K8V8 smoke | the recorder's `(N, D, 256)` argmin tensor; the full run uses 2-3-bit keys and needs under 0.1 GB |
| smoke time | 10-25 min | 3 cells x 2 sentences x 4 generations, plus model load |
| full run time | 3-6 h per cell at MAXPG=100 on an A100-40GB | 100 sentences x (3 TQ + fp16 + fp16 divergence pass); 3 cells = 9-18 h; scoring adds 20-40 min per cell |
| disk | about 2-3 GB | wavs ~0.2 GB per cell; divergence CSVs ~0.5 GB per cell; scores negligible |
| CPU summary | seconds | `tools/summarize_keyfix.py` |

Timing is scaled from the stage table in `scripts/gpu/README.md` (6 configs x
100 sentences = 4-8 h per seed on the same script).

## Outputs

| file | content |
|---|---|
| `results/qwen_keyfix_<cell>_seed<s>.csv` | per (layer, position, config) divergence rows, with `key_group`, `pre_rope` columns |
| `models/Qwen3-TTS/benchmarks/outputs/keyfix_<cell>/*.wav` | compressed wavs, name segments `_g=32` / `_pre` / `_pp=prompt` / `_pq=k2` (`_pq=m3` for min-max; the number is the effective prompt bits); fp16 shared |
| `results/qwen_keyfix_<cell>_scores.csv` | Whisper large-v3 CER/WER, WavLM spk_sim vs fp16 (`SCORE=1`, default) |
| `results/keyfix_summary.md` | per cell x config objective + divergence tables and the per-layer attn_js profile for K2V2@0 |

Config labels in the scores CSV are `K2V2@0+g32`, `K2V2@0+pre`, `K2V2@0+g32+pre`, and for the prefix codebook `+pk2`, `+pk3`, `+pm3`
(appended after the other flag labels);
all existing analysis tools treat them as ordinary extra configs.

## Bit accounting

Group scaling stores two fp16 scalars per coordinate per 32 tokens, which is
1.0 bit per key element on top of the packed codes. `memory_report()` and the
`theoretical_ratio` column include it through the shared
`compressed_bytes_per_token` helper, so a K2V2+g32 cache is reported at 3.125
bits per key element, not 2. State this in the thesis. OmniKVQuant's released
code stores the same two scalars per coordinate per group but in MXFP4 (4-bit
E2M1 plus one shared 8-bit exponent per 32 coordinates), i.e. 0.27 bits per key
element; the paper's "bits/elem" figures (2.258 / 2.625) are the average over
keys and values (keys 2.39 / 3.125, values 2.125). Our fp16 metadata is the
best case for their method, which is what a mechanism test wants.

Prefix codebook metadata: quantizing the prompt with fitted levels stores
`meta_scalars` fp16 scalars per coordinate for the whole prompt, amortized as
`meta_scalars * 16 / P` bits per prompt key element (P = prompt length), where
`meta_scalars` is L = 2^b for k-means and 2 for minmax. The norm is excluded, as
for every other cell, so the pooled figures in the cell table (3.04 / 3.42 /
3.11) are this formula with the per-vector norm left out, not numbers read from
the cache; `memory_report()` includes the norm and gives slightly higher ones.

## Smoke history

- 2026-11-06, first run on the SageMaker box: `plain` and `g32` clean (115266
  rows each); `pre` printed PASS but had 46397 skipped measurements and only
  22472 rows. Cause: the divergence recorder kept one rotary-table buffer per
  attention class and treated every layer's call as a new step, so on each
  decode step only layer 0 was recorded. The cache's own `_record_rope` had
  the per-layer guard; the recorder did not. Fixed (same-step tables are
  reused; `--step-stride` no longer starves the buffer), the validator now
  requires equal row counts across cells and zero measurement errors, and
  `14_qwen_keyfix.sh` aborts a cell that reports measurement errors. The
  `sox: not found` banner at start-up is torchaudio probing a backend it
  does not use here; harmless.

## Getting the results off the box

The raw divergence CSVs are ~750 MB each (15 M rows) and the JupyterLab file
browser truncates downloads at 24 MiB. Reduce first, then split:

```
$PY tools/reduce_keyfix_div.py            # -> results/keyfix_reduced/*_bysent.csv, *_bypos.csv, *_bystep.csv, *_bytime.csv, *_bylen.csv
ls results | grep keyfix | grep -v seed0 | sed s,^,results/, > keyfix_list.txt
ls results/keyfix_reduced | sed s,^,results/keyfix_reduced/, >> keyfix_list.txt
ls logs | grep keyfix | sed s,^,logs/, >> keyfix_list.txt
tar czf keyfix_results.tgz -T keyfix_list.txt
split -b 20m keyfix_results.tgz keyfix_part_
```

On the laptop: `cat keyfix_part_aa keyfix_part_ab > keyfix_results.tgz`,
`tar xzf ... -C results/keyfix`, then
`python tools/analyze_keyfix.py --root results/keyfix --out results/keyfix`
writes `keyfix_analysis.md` and `keyfix_layers.{png,pdf}`.

## Results, seed 0, 50 sentences (2026-11-10)

Paired: same sentences and seed in all four cells, plain cell re-run under the
same protocol and torch build (2.6.0+cu124).

| config | plain | g32 | pre | g32pre |
|---|---|---|---|---|
| K2V2 collapse % | 100 | 100 | 100 | 100 |
| K2V4 collapse % | 100 | 100 | 100 | 100 |
| K3V3 collapse % | 100 | 28 | 100 | 6 |
| K3V3 CER | 0.99 | 0.31 | 0.92 | 0.13 |
| K3V3 spk_sim | 0.73 | 0.93 | 0.75 | 0.96 |
| K2V2 talker attn_js | 0.232 | 0.078 | 0.236 | 0.061 |
| K3V3 talker attn_js | 0.102 | 0.015 | 0.102 | 0.011 |

What it says

- Group scaling (E1) lowers attention divergence in 50/50 sentences at every
  bit width (Wilcoxon p < 1e-14) and rescues 3-bit keys (28 % collapse vs
  100 %), but 2-bit keys still collapse in every sentence. The paper's fix
  transfers partially.
- Pre-RoPE (E2) alone changes nothing: per-layer curves sit on top of plain.
  RoPE is not the cause of the range drift.
- g32 vs g32pre at K3V3: 14 vs 3 collapses, disjoint sentence sets (no sentence
  collapses in both). Discordant-pair test p = 0.013, but the disjointness
  says collapse at this operating point is a per-run sampling event, not a
  sentence property. Needs seeds 1-2 before claiming the stack helps.
- Within g32 K3V3, per-sentence mean attn_js does not separate collapsed from
  surviving sentences (0.0146 vs 0.0149, p = 0.3; same at layer 5). The
  collapse is not predicted by average divergence; it is a step event. A
  per-step max/95th-percentile reduction would test that directly (raw CSVs
  are still on the box).
- Per-layer (talker only): plain peaks at layer 0 (0.66 at K2V2); g32 cuts
  layer 0 by 11x but layer 5 by only 2.4x, so layer 5 (0.225) becomes the
  peak of both g32 cells. Layer 5 is the group-scaling-resistant layer.
- Along the sequence: plain divergence grows with decode position (layer 0:
  0.26 at <64 steps to 0.47 at 512+), the drift signature; under g32 it falls
  with position (0.020 to 0.007). Group scaling removes the length dependence.

## Results, prompt protection, seed 0, 50 sentences (2026-11-17)

Cells `pp` (whole prefill kept in fp16, `protect_prefix=-1`) and `g32pp`
(same plus group scaling), configs fp16/K2V2/K3V3, same sentences and seed as
above, so every comparison is paired.

| config | plain | g32 | pp | g32pp |
|---|---|---|---|---|
| K2V2 collapse % | 100 | 100 | 100 | 0 |
| K2V2 CER | 1.04 | 0.95 | 1.00 | 0.003 |
| K2V2 spk_sim | 0.69 | 0.73 | 0.82 | 0.992 |
| K3V3 collapse % | 100 | 28 | 58 | 0 |
| K3V3 CER | 0.99 | 0.31 | 0.74 | 0.004 |
| K3V3 spk_sim | 0.73 | 0.93 | 0.92 | 0.992 |
| K2V2 talker attn_js | 0.248 | 0.075 | 0.164 | 0.019 |
| K2V2 talker attn_top1 | 0.42 | 0.62 | 0.71 | 0.89 |
| K3V3 talker attn_js | 0.110 | 0.014 | 0.073 | 0.003 |
| K2V2 layer-5 attn_js | 0.546 | 0.225 | 0.440 | 0.036 |

fp16 reference: CER 0.005. g32pp at K2V2 is indistinguishable from fp16 on
every audio metric.

What it says

- Neither fix alone rescues 2-bit keys; the two together rescue them
  completely (0/50 collapses, CER 0.003, spk_sim 0.992 at K2V2, and the same
  at K3V3). Prompt protection alone is the weaker single fix (K3V3 collapse
  58% vs 28% for g32).
- The two fixes act on different keys. g32 repairs the generated keys (their
  per-coordinate range drifts, section "Gaussianity diagnostic" in the
  handoff); the prompt keys do not drift but carry 23-73% of the attention
  mass, and a 2-bit code of them with the fixed codebook is what the g32 cells
  were still missing. Layer 5, the g32-resistant layer, drops from 0.225 to
  0.036 once the prompt is exact, which matches the diagnostic's reading that
  its low-rank queries point at the prompt.
- Mean divergence still does not predict collapse across cells: pp at K3V3
  has lower attn_js than plain (0.073 vs 0.110) and higher top-1 agreement
  (0.80 vs 0.58) yet collapses in 58% of sentences; g32 at K3V3 with top-1
  0.79 collapses in 28%. Collapse remains a threshold event on something the
  means do not capture.
- Bit accounting (section 8 of `keyfix_analysis.md` once the `_bylen.csv`
  files are reduced). The prefix is 44-67 keys and these LibriSpeech sentences
  are short (100-770 keys in total), so at K2 the g32pp cell costs about 5
  bits per key element pooled over the 50 sentences (2 code + 1 group metadata
  + the fp16 prefix), falling toward 3.7 for 1000-key sequences. The honest
  comparison is therefore against plain 4-bit keys at the same budget, and
  the honest claim is a mechanism claim: the 2-bit failure lives in the prompt
  keys plus the drift of the generated keys, not in the generated keys alone.
- Next cells to separate "prompt" from "sink" and to shrink the cost:
  `protect_prefix` 4 / 16 / 32 (fixed N) with g32 at K2V2. If N = 4 already
  rescues, it is an attention-sink effect; if only the full prompt does, the
  reference audio keys are the carrier.

