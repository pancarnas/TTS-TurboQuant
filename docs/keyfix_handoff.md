# Key-fix experiment: handoff (2026-11-16)

Read this first when picking up `exp/keyfix-group-prerope` on a new machine.
Run book with commands: `docs/keyfix_experiment.md`. Background and option
list: `docs/followup_omnikvquant_plan.md`. Shared summary doc:
https://claude.ai/artifact/H9djMBo2H4ZWLG5y5G6Azg

## State in one paragraph

Four cells (plain, g32, pre, g32pre) ran on Qwen3-TTS, 50 LibriSpeech-PC
sentences, clone mode, seed 0, configs fp16/K2V2/K2V4/K3V3 with rw=0, on the
SageMaker box (NVIDIA L4, torch 2.6.0+cu124). Group key scaling (g32, the
OmniKVQuant fix, +1 bit/elem metadata) rescues 3-bit keys (collapse 100% ->
28%, spk_sim 0.73 -> 0.93) but 2-bit keys collapse in every sentence in every
cell. Pre-RoPE quantization alone changes nothing (RoPE is not the cause).
g32pre at K3V3 shows 6% collapse vs 28% for g32, but the collapsed sentence
sets are disjoint and per-sentence divergence does not predict collapse, so
that difference is unproven until seeds 1-2 run. Talker-only per-layer
divergence: plain peaks at layer 0 (the old "layer-5 hotspot" was a
code-predictor artifact), g32 cuts layer 0 by 11x but layer 5 by only 2.4x,
so layer 5 is the scaling-resistant layer. Plain divergence grows with decode
position, g32 divergence does not (the drift signature).

## Diagnostic results (15_qwen_keygauss.sh, 20 sentences, 2026-11-17)

Table-1 analogue (32-token windows; Gaussian floor 96.9% / 0.215 / 0.149
reproduces the paper's reference row): keys post-RoPE retain 16.6% variance
(W1 0.899, KS 0.545) vs the paper's 30.4% on Qwen2.5-Omni; prompt keys 32.1%,
generated keys 12.7%, pre-RoPE keys 9.1% (RoPE spreads coordinates, so undoing
it cannot help), values 32.0%. offset (energy of per-coordinate mean) 0.70;
drift (variance between windows) 0.44 overall, -0.04 on prompt keys, 0.49 on
generated keys. Key relmse at 2 bits: fixed 0.116 / g32 0.030 / static affine
(10 calib sentences) 0.057; at 3 bits 0.034 / 0.005 / 0.019 -> affine alone
cannot rescue (3-bit fixed at 0.034 collapses). Layer 5: q effective rank 7.2,
k 6.8, only 2.1% of key variance in the top-8 query directions (iso 6.25%) ->
query-aware rotation case; low-q-rank layers: 3,4,5,10,15,20,21,22,27.
Attention mass on the prompt (44-67 tokens, carries the text) is 23-73% per
layer -> prompt protection in fp16 is the cheapest mechanism test.
Outputs on the box: results/keygauss_stats.csv, _layers.csv, _summary.md.

## Results on disk

Laptop (not tracked by git): `results/keyfix/` = the downloaded archive
(`results/qwen_keyfix_<cell>_scores.csv`, `results/keyfix_reduced/*_bysent.csv`
and `*_bypos.csv`, `results/keyfix_summary.md`, `logs/`) plus the analysis
output `results/keyfix/keyfix_analysis.md` and `keyfix_layers.{png,pdf}` from
`python tools/analyze_keyfix.py --root results/keyfix --out results/keyfix`.

Box (`~/TTS-TurboQuant`): everything above plus the raw divergence CSVs
`results/qwen_keyfix_<cell>_seed0.csv` (~750 MB, 3.7 M rows each) and the wavs
under `models/Qwen3-TTS/benchmarks/outputs/keyfix_<cell>/`.

## Next steps, in order

1. Per-step maximum (done 2026-11-17: `tools/reduce_keyfix_div.py` now emits, per
   (sentence, layer, config), `_bystep.csv` with max, p95 and threshold fractions, and
   `_bytime.csv`, the per-sentence time course). Run on the box (CPU, minutes). Tests the
   "collapse is a single-step event" hypothesis.
1b. Gaussianity diagnostic (OmniKVQuant Table 1 analogue + what fix to try),
   ~10 GPU minutes, 20 sentences, no quantization on the path:
   `MAXPG=20 bash scripts/gpu/15_qwen_keygauss.sh` -> `results/keygauss_stats_summary.md`
   (retained variance / W1 / KS per 32-token window for keys, pre-RoPE keys and
   values vs a true Gaussian; offset vs drift; relmse of fixed Lloyd-Max vs g32
   min-max vs a static per-coordinate affine map; per-layer query/key effective
   rank, key variance in the top-8 query directions, attention mass on the first
   64 keys and on the prompt). Smoke with MAXPG=2 first. Download the three
   `results/keygauss_stats*` files (small).
2. Seeds. On the box, ~5 GPU hours:
   `CELLS="g32 g32pre" CFGS="fp16,K3V3@0" SEEDS="1 2" MAXPG=50 nohup bash scripts/gpu/14_qwen_keyfix.sh > logs/keyfix_seeds.log 2>&1 &`
   Settles whether g32pre's 6% is real. Then re-run the reducer, re-download,
   re-run `tools/analyze_keyfix.py`.
2b. Prompt protection: keep the first P (prompt) positions of the cache in
   fp16 (config field + cache change; library edit, ask first). One cell each
   at K2V2 and K3V3, 50 sentences, seed 0. Honest bit accounting:
   (P*16 + (S-P)*b)/S per element.
3. Query-aware key rotation (the 2-bit candidate, from OSCAR arXiv:2605.17757
   and NOVA-KV arXiv:2608.04074): calibrate the query second moment per
   (layer, KV head) on ~20 sentences, rotate keys into its eigenbasis instead
   of the random rotation. Needs an optional rotation override in
   `turboquant/compressors_v3.py` (ask before editing the library) and a
   calibration tool. Also worth trying: keep the first 64 prompt tokens in
   fp16 (sink protection) as a cheap cache-only change.
4. Optional: VALL-E-X with `--key-group 32` to confirm it does not hurt the
   model that was already fine.

## Pitfalls on the SageMaker box

- The chat client turns quotes into curly quotes and strips `*`. Type
  commands by hand, one per line, no trailing comments, no wildcards; build
  file lists with `ls | grep | sed` instead.
- Never run `uv sync`: it strips every package not in pyproject (pandas,
  whisper, qwen_tts) and upgraded torch to 2.15. `scripts/gpu/_ensure_env.sh`
  reinstalls what is missing; torch must stay `2.6.0+cu124`
  (`uv pip install --python .venv/bin/python torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124`).
- Box restarts kill `nohup` jobs. Every script resumes per sentence; relaunch
  with the same command.
- JupyterLab file download truncates at 24 MiB. Reduce first, then
  `split -b 20m` and `cat` the parts on the laptop (run book has the lines).
- `echo ======` in zsh triggers `=cmd` expansion; quote such strings.
- Summary tables pasted into chat render as `[object Object]`; use
  `tr \| : < results/keyfix_summary.md`.
- The divergence recorder also hooks the code predictor (`Qwen3TTSAttention`,
  pos < 33, 16x rows in layers 0-4). `tools/summarize_keyfix.py` and
  `tools/reduce_keyfix_div.py` filter on pos >= 33; the recorder fix (skip that
  class in `make_patch`) is proposed, not applied.
