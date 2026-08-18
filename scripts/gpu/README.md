# Running the full KV-quant campaign on a plain GPU machine

End-to-end reproduction of the VALL-E-X vs Qwen3-TTS KV-cache-quantization
experiments — generation, objective metrics (Whisper CER/WER + WavLM speaker
similarity), mechanism diagnostics (teacher-forced PPL, attention divergence),
and all thesis tables/figures — on a single rented GPU box. No cluster, no
Grid Engine; every stage is a plain bash script run from the repo root.

This mirrors the clean-clone campaign documented in
[`docs/results_provenance.md`](../../docs/results_provenance.md): zero-shot
voice cloning on LibriSpeech-PC, English preset `librispeech_1.npz`, seed 0
for the config grid, seeds 0/1/2 for the headline runs.

## Hardware / environment

| | requirement |
|---|---|
| GPU | 1× NVIDIA, **24 GB VRAM recommended** (16 GB works with `NSHARDS=1`) |
| CUDA | driver for CUDA 12.x (`nvidia-smi` to check; scripts install cu124 torch) |
| Disk | ~150 GB free (wavs dominate: grid ≈ 2×5600 clips, seeds ≈ 2×1500) |
| OS | Linux with `apt` (Ubuntu/Debian images on Lambda/RunPod/vast.ai are fine) |
| Network | needed for setup + first run of each stage (pip, HF models, Whisper) |

Models downloaded on first use: VALL-E-X checkpoint, EnCodec, Vocos,
Qwen3-TTS-1.7B, Whisper large-v3, WavLM x-vector (~15 GB total, cached in
`~/.cache` / `./checkpoints`).

## TL;DR

```bash
git clone <this-repo> && cd TTS-TurboQuant
bash scripts/gpu/00_setup.sh                       # once
nohup bash scripts/gpu/run_all.sh > logs/run_all.log 2>&1 &   # everything else
tail -f logs/run_all.log
```

Results land in `results/` (CSV/markdown tables) and `results/figures/` (PNGs).

## Stage by stage

Run them in order; each is independently restartable. Rough wall-clock is for
a single A100-40GB with `NSHARDS=3` — halve/double to taste.

| # | script | what it does | output | time |
|---|---|---|---|---|
| 0 | `00_setup.sh` | venv, sox, project packages, metrics deps, cu124 torch | `.venv/` | 10 min |
| 1 | `01_fetch_data.sh` | seed-tts-eval + ellav_hard + **LibriSpeech-PC** (the clone-mode set) + validation | `data/` | 10 min |
| 2 | `10_vallex_grid.sh` | VALL-E-X **28-config grid** (K,V∈{4,3,2} × rw∈{0,64,128} + fp16), clone mode, seed 0, at pl0 **and** pl2; records attention divergence in the same pass | `outputs/grid_cl_pl{0,2}/`, `results/grid_cl_pl{0,2}_div_shard*.csv` | 12–24 h/pl |
| 3 | `11_vallex_seeds.sh` | 5 headline configs (fp16, K4V4@0, K4V2@0, K3V3@0, K2V2@0) × **seeds 0,1,2**, pl0 + pl2 → error bars | `outputs/seed_cl_pl{0,2}/`, `results/seed_cl_pl{0,2}_div_shard*.csv` | 6–12 h/pl |
| 4 | `12_vallex_ppl.sh` | teacher-forced NLL/PPL/KL/top1-agree over ground-truth EnCodec tokens, 9 rw0 configs (mechanism: per-step damage vs free-running collapse) | `results/vallex_ppl_cl.csv`, `results/vallex_ppl_div_cl.csv` | 2–4 h |
| 5 | `13_qwen_divergence.sh` | Qwen3-TTS counterfactual divergence + wavs, 6 configs × seeds 0,1,2 (the cross-model contrast) | `results/qwen_div_seed{0,1,2}.csv`, Qwen wavs | 4–8 h/seed |
| 6 | `20_score.sh` | **objective metrics** on every wav: Whisper large-v3 CER + WER vs ground-truth text, WavLM speaker cosine vs the same sentence's fp16 wav | `results/*_scores.csv` | 4–8 h |
| 7 | `30_analyze.sh` | tables + stats + figures (CPU-only; also runs on a laptop given the CSVs) | see below | 10–30 min |

### What `30_analyze.sh` produces

| artifact | meaning |
|---|---|
| `results/clone_seed_errorbars.md` (+ png) | per-config CER/WER mean ± std across seeds, pl0 vs pl2 |
| `results/master_cl_pl0.csv` | one row per config: CER/WER/collapse%/spk_sim + PPL/KL + attn_js/cos_k/cos_v |
| `results/cross_model_ci.md` | Qwen vs VALL-E matched table, WER ± sd + collapse % with 95% Wilson CIs |
| `results/clone_rw_significance.md` | does the residual window matter? paired Wilcoxon (Holm) per bit-pair |
| `results/combined_clone_summary.{csv,md}` | per (model, group, config): CER/WER/collapse/spk_sim/cos_k/cos_v |
| `results/{vallex,qwen}_layer_summary*.csv` | per-layer attention divergence (for the layer-profile figure) |
| `results/figures/clone_kv_pl_{cer,wer}.png` | K×V heatmaps per protection level |
| `results/figures/clone_tradeoff_{cer,wer}.png` | quality vs realised compression, Pareto frontier |
| `results/figures/layer_profile_attnjs.png` | MHA-flat (VALL-E) vs GQA layer-hotspot (Qwen) profile |
| `results/figures/cross_model_collapse.png` | collapse-rate contrast at matched bits |

## Knobs

Environment variables read by the generation scripts:

- `NSHARDS` (default 2) — parallel workers sharing the one GPU. 3–4 on a
  40 GB card, 1 if you hit CUDA OOM. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  is already set by the runner to curb fragmentation.
- `MAXPG` (default 100) — sentences drawn from LibriSpeech-PC per group.
  Sentence draws are **paired across configs and seeds**, so rerunning with a
  larger `MAXPG` regenerates the earlier wavs byte-identically and adds new ones.

## Restart / resume behaviour

- **Generation** (stages 2–3): deterministic paired seeds — safe to rerun; the
  post-hoc scorer skips already-scored rows, so duplicates cost only compute.
- **PPL, Qwen divergence, scoring** (stages 4–6): pass `--resume` and skip
  completed rows — just rerun the same script after an interruption.
- Prefer rerunning the failed stage script over `run_all.sh`.

## Protocol notes (do not change these silently)

- **Preset must be `librispeech_1.npz`** (English). `alan.npz` is Japanese and
  makes VALL-E-X synthesize English cross-lingually, inflating WER ~5× — that
  mistake invalidated an entire earlier campaign
  (see `docs/eddie_run_state.md`).
- **Clone mode** (`--voice-mode clone`) with `librispeech_pc` is the standard
  zero-shot protocol: per-item reference voice + real ground-truth recording.
- **pl0** (no protected layers) is the deployment operating point used for the
  headline numbers; pl2 is the ablation.
- spk_sim is measured against the *fp16 wav of the same sentence*, so fp16
  rows legitimately have a blank spk_sim.

## Troubleshooting

- **CUDA OOM during generation** — `NSHARDS=1 bash scripts/gpu/10_vallex_grid.sh`.
- **WavLM/speaker-sim crashes with a cuDNN error** — a stray tensorflow install;
  `pip uninstall -y tensorflow tensorflow-cpu tf-keras` (setup already does this).
- **`sox: not found`** — `sudo apt-get install -y sox libsox-dev`.
- **torchaudio/libcudart mismatch** — reinstall torch matching your driver:
  `pip install --force-reinstall torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124`.
- **HF download failures** — make sure `HF_HUB_OFFLINE` is unset; rerun the
  stage (everything resumes).
- Machine much smaller than an A100? Cut scope first via `MAXPG=50`, and drop
  pl2 from `10_vallex_grid.sh`/`11_vallex_seeds.sh` (edit `for PL in 0 2` →
  `for PL in 0`) — pl0 alone supports every headline claim except the
  protection ablation.
