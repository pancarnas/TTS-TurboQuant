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

# 2. full run, Qwen, seed 0, three cells (9-18 h at MAXPG=100; see knobs)
nohup bash scripts/gpu/14_qwen_keyfix.sh > logs/keyfix.log 2>&1 &

# 3. read results/keyfix_summary.md (re-runnable any time, also mid-run)
python tools/summarize_keyfix.py --results results \
    --plain-div results/qwen_div_seed0.csv --plain-scores results/qwen_seed_scores.csv
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
- CSV rows carry `key_group` / `pre_rope` with the cell's values

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
| `models/Qwen3-TTS/benchmarks/outputs/keyfix_<cell>/*.wav` | compressed wavs, name segments `_g=32` / `_pre`; fp16 shared |
| `results/qwen_keyfix_<cell>_scores.csv` | Whisper large-v3 CER/WER, WavLM spk_sim vs fp16 (`SCORE=1`, default) |
| `results/keyfix_summary.md` | per cell x config objective + divergence tables and the per-layer attn_js profile for K2V2@0 |

Config labels in the scores CSV are `K2V2@0+g32`, `K2V2@0+pre`, `K2V2@0+g32+pre`;
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
