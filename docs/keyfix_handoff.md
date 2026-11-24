# Key-fix experiment: handoff (2026-11-23)

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

**Prompt protection (2026-11-17, cells pp / g32pp, same 50 sentences, seed 0,
configs fp16/K2V2/K3V3):** keeping the prefill keys and values in fp16 and
group-scaling the rest rescues 2-bit keys completely: g32pp K2V2 = 0/50
collapses, CER 0.003, spk_sim 0.992 (fp16: CER 0.005); K3V3 the same. Neither
fix alone does it (pp alone: K2V2 100%, K3V3 58% collapse; g32 alone: K2V2
100%, K3V3 28%). Talker attn_js at K2V2: plain 0.248, g32 0.075, pp 0.164,
g32pp 0.019; layer 5 falls from 0.225 (g32) to 0.036 (g32pp). Cost: the prefix
is 44-67 keys and the sentences are 100-770 keys, so g32pp K2 is about 5 bits
per key element pooled over this set (3.7 at 1000 keys). Full table and
reading in `docs/keyfix_experiment.md`, "Results, prompt protection".

**Prefix codebook (2026-11-19/20, cells pk2 / pk3 / pm3 / pm4, same 50
sentences, seed 0, all over g32pre generated keys):** quantizing the prompt
keys at prefill with levels fitted on the prompt. k-means levels (pk2, pk3)
collapse 100% like g32pre although pk3 has the lowest key error of any cell;
one min-max window at 3 bits (pm3) collapses 36%; at 4 bits (pm4) K2V2
collapses 1/50 (CER 0.034, spk_sim 0.954) and K3V3 0/50 (CER 0.007, WER
0.013, spk_sim 0.959; fp16 CER 0.005 / WER 0.014). Pooled key bits (section
8): pm4 K2V2 3.30 (3.09 at 1000 keys), K3V3 4.11 (4.03), against g32pp 5.44
/ 6.25. So K3V3+pm4 is the fully quantized cache at fp16 intelligibility and
K2V2+pm4 the aggressive point. The one pm4 collapse is sentence 4, a runaway
to the generation cap that also defeats 16 of the other 20 quantized
settings; pm4 at K3V3 passes it. Reading: the prompt KEYS carry the collapse
and what matters is keeping the per-coordinate range exact, not the MSE
("range, not shape"). Prompt values at 2-3 bits do not cause collapse but
likely cost the 0.04 of spk_sim against g32pp. Table and reading in
`docs/keyfix_experiment.md`, "Results, prefix codebook".

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
(`results/qwen_keyfix_<cell>_scores.csv`, `results/keyfix_reduced/*_bysent.csv`,
`*_bypos.csv`, `*_bystep.csv`, `*_bytime.csv`, `*_bylen.csv`,
`results/keyfix_summary.md`, `logs/`) plus the analysis
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
2b. Prompt protection: DONE 2026-11-17 (`TurboQuantConfig.protect_prefix`,
   -1 = whole prefill, N > 0 fixed; cells `pp` and `g32pp` in 03/14 scripts;
   wav segment `_pp=prompt` / `_pp=N`, CSV column `protect_prefix`, config
   label `+pp` / `+ppN`). Result: g32pp rescues K2V2 completely (see "State").
   Prefix codebook cells pk2 / pk3 / pm3 / pm4: DONE 2026-11-20, reduced,
   downloaded and analysed 2026-11-23 (see "State"). Remaining optional cell,
   zero code, ~1.5 GPU h: `CELLS="pm4" CFGS="fp16,K3V4@0" MAXPG=50` to see
   whether prompt values at 4 bits close the spk_sim gap to g32pp (0.96 vs
   0.99). Cells B (fp16 first 4 tokens) and D (dense-and-sparse) of
   `docs/keyfix_prefill_proposal.md` are no longer needed for collapse.
2c. Prefix sweep (next GPU job, ~2 h per N): `protect_prefix` 4 / 16 / 32 with
   g32 at K2V2 separates an attention-sink effect (N = 4 enough) from the
   reference-audio keys (only the full prompt works) and finds the cheapest
   rescue. Needs a `g32ppN` cell in `14_qwen_keyfix.sh` (one line each, copy
   the `g32pp` case with `--protect-prefix N`); the wav suffix, scorer and
   summariser already handle fixed N. Also a matched-budget control: plain
   K4V2 and K4V4 at rw=0 from the main grid, so the thesis can say what
   ~4-5 effective key bits buy without the mechanism.
3. Query-aware key rotation (the 2-bit candidate, from OSCAR arXiv:2605.17757
   and NOVA-KV arXiv:2608.04074): calibrate the query second moment per
   (layer, KV head) on ~20 sentences, rotate keys into its eigenbasis instead
   of the random rotation. Needs an optional rotation override in
   `turboquant/compressors_v3.py` (ask before editing the library) and a
   calibration tool. Lower priority now that g32pp rescues K2: its value is
   a metadata-free alternative to protecting the prompt.
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
