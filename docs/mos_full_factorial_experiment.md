# Full-factorial 39-speaker objective campaign — runbook

Extend the 39-speaker objective study to **all 28 configs × both protection levels
(pl0, pl2) × all seeds (0,1,2)** on the same 195-sentence set (`mos_idx_list.txt`),
so every claim (quality ladder, 3-bit≈fp16, rw-inertness, protection/Pareto,
attention mechanism) has seed error bars *and* the pl comparison. Independent of the
MOS/SMOS listening package (pl0/rw0/seed0, already built + hosted).

**Scale:** 28 × 2 × 3 × 195 ≈ **32,760 gens**; 5,460 already done (pl0/seed0). New ≈
**27,300** — multi-night, chunk with `--resume`. Divergence recorded **only at seed 0
per pl** (the recorder has no `seed` column, so multi-seed divergence can't be
separated); seed robustness is carried by the per-seed quality metrics.

**Skip pl0/seed0** — `grid_clone_mos` already has it; reused as `grid_pl0`'s seed 0.

```bash
# the 28-config string (reused in every generation command)
CFGS="fp16,K2V2@0,K2V2@128,K2V2@64,K2V3@0,K2V3@128,K2V3@64,K2V4@0,K2V4@128,K2V4@64,K3V2@0,K3V2@128,K3V2@64,K3V3@0,K3V3@128,K3V3@64,K3V4@0,K3V4@128,K3V4@64,K4V2@0,K4V2@128,K4V2@64,K4V3@0,K4V3@128,K4V3@64,K4V4@0,K4V4@128,K4V4@64"
REMOTE=s2801778@eddie.ecdf.ed.ac.uk:/exports/chss/eddie/ppls/groups/slpgpustorage/users/s2801778/tts_cw/TTS-TurboQuant
```

`run_vallex_perm.sh` positional args: `N MAXPG group TAG CFGS PL SUBDIR SEEDS PRESET RECDIV VOICEMODE IDXFILE`.

---
## Part A — Directory reorg

**Laptop** (archive old, clean 39-speaker `results/`):
```bash
mv results "results_archive_8spk_$(date +%Y%m%d)"
mkdir -p results/clone results/figures
cp -r results_archive_8spk_*/final_39speaker results/
cp results_archive_8spk_*/clone/{grid_clone_mos_scores,master_clone_mos,vallex_ppl_mos,vallex_layer_summary_mos}.csv results/clone/
cp results_archive_8spk_*/qwen_layer_summary.csv results_archive_8spk_*/qwen_seed_scores.csv results/ 2>/dev/null || true
cp results_archive_8spk_*/vallex_grid_clone_mos_div.csv results/ 2>/dev/null || true   # pl0/seed0 divergence, if kept local
```

**Eddie** (group 39-speaker audio; new runs land here):
```bash
mkdir -p models/VALL-E-X/benchmarks/outputs/mos_39spk
mv models/VALL-E-X/benchmarks/outputs/grid_clone_mos models/VALL-E-X/benchmarks/outputs/mos_39spk/grid_pl0
```

---
## Step 0 — SMOKE TEST (Eddie — GPU steps MUST be qsub'd; login node has no GPU)
```bash
head -2 mos_idx_list.txt > smoke_idx.txt
# (1) GEN — GPU job
qsub -N smoke_gen -l h_rt=1:00:00 -pe sharedmem 2 scripts/eddie/eddie_run.sh \
  bash scripts/eddie/run_vallex_perm.sh 2 999999 librispeech_pc smoke_pl2 \
  "fp16,K4V4@0,K2V2@0" 2 mos_39spk/smoke_pl2 "0,1" librispeech_1.npz div clone smoke_idx.txt
# wait: qstat until empty, then check
ls models/VALL-E-X/benchmarks/outputs/mos_39spk/smoke_pl2/*.wav | wc -l   # expect 12
ls results/smoke_pl2_div_shard*.csv                                       # div recorded
# (2) SCORE — GPU job
qsub -N smoke_score -l h_rt=1:00:00 -pe sharedmem 2 scripts/eddie/eddie_run.sh \
  python tools/score_wav_dir.py --audio-dir models/VALL-E-X/benchmarks/outputs/mos_39spk/smoke_pl2 \
  --data-dir data --device cuda --out results/clone/smoke_pl2_scores.csv --resume
# wait, then (3) AGGREGATE — CPU (login node ok; torch imports without a GPU)
python tools/aggregate_seeds.py --scores "smoke=results/clone/smoke_pl2_scores.csv" \
  --out /tmp/smoke_seed.md --img /tmp/smoke_seed.png
```
If a GPU job shows 0 wavs, read `logs/smoke_gen.o<jobid>` / `logs/smoke_pl2_shard0.log`.
**PASS** = 12 wavs `vallex_librispeech_pc_<idx>_sampling_s{0,1}_<cfg>.wav`; non-empty div
shard; 12 score rows with valid CER/WER; `aggregate_seeds` writes md+png. Then clean up:
```bash
rm -rf models/VALL-E-X/benchmarks/outputs/mos_39spk/smoke_pl2 results/smoke_pl2_div_shard*.csv \
       results/clone/smoke_pl2_scores.csv smoke_idx.txt
```

---
## Part B — Generation (Eddie GPU, 3 new batches; submit as jobs)
```bash
# 1) pl0, seeds 1,2  (no divergence)   -> ~10,920 gens
qsub -N vx_pl0_s12 -l h_rt=24:00:00 -pe sharedmem 8 scripts/eddie/eddie_run.sh \
  bash scripts/eddie/run_vallex_perm.sh 8 999999 librispeech_pc grid_mos_pl0_s12 \
  "$CFGS" 0 mos_39spk/grid_pl0 "1,2" librispeech_1.npz "" clone mos_idx_list.txt

# 2) pl2, seed 0  (WITH divergence)    -> ~5,460 gens
qsub -N vx_pl2_s0 -l h_rt=24:00:00 -pe sharedmem 8 scripts/eddie/eddie_run.sh \
  bash scripts/eddie/run_vallex_perm.sh 8 999999 librispeech_pc grid_mos_pl2_s0 \
  "$CFGS" 2 mos_39spk/grid_pl2 "0" librispeech_1.npz div clone mos_idx_list.txt

# 3) pl2, seeds 1,2  (no divergence)   -> ~10,920 gens
qsub -N vx_pl2_s12 -l h_rt=24:00:00 -pe sharedmem 8 scripts/eddie/eddie_run.sh \
  bash scripts/eddie/run_vallex_perm.sh 8 999999 librispeech_pc grid_mos_pl2_s12 \
  "$CFGS" 2 mos_39spk/grid_pl2 "1,2" librispeech_1.npz "" clone mos_idx_list.txt
```
Monitor: `qstat`; `ls outputs/mos_39spk/grid_pl2/*.wav | wc -l`; `tail -f logs/grid_mos_pl2_s0_shard0.log`.
Re-submit any job that hits `h_rt` — `--resume` on scoring + paired seeds make it safe.

---
## Part C — Scoring (Eddie GPU — qsub, one job per pl dir)
```bash
for sub in grid_pl0 grid_pl2; do
  qsub -N score_$sub -l h_rt=12:00:00 -pe sharedmem 4 \
    -v OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1,MKL_NUM_THREADS=1 \
    scripts/eddie/eddie_run.sh \
    python tools/score_wav_dir.py --audio-dir models/VALL-E-X/benchmarks/outputs/mos_39spk/$sub \
      --data-dir data --device cuda --out results/clone/${sub}_mos_scores.csv --resume
done
```
→ `grid_pl0_mos_scores.csv`, `grid_pl2_mos_scores.csv` (each 28 cfg × 3 seeds × 195).
Verify: `python tools/check_mos_campaign.py --scores results/clone/grid_pl2_mos_scores.csv`.

---
## Part D — Aggregation + masters + figures

**Eddie (torch):**
```bash
# per-pl seed error bars
for pl in pl0 pl2; do
  python tools/aggregate_seeds.py --scores "VALL-E-X=results/clone/grid_${pl}_mos_scores.csv" \
    --out results/clone_seed_errorbars_${pl}.md --img results/figures_seed/${pl}.png
done
# pl2 divergence -> master (pl0 master already exists as master_clone_mos.csv)
awk 'FNR==1 && NR!=1{next}1' results/grid_mos_pl2_s0_div_shard*.csv > results/vallex_grid_pl2_div.csv
python tools/merge_metrics.py --scores results/clone/grid_pl2_mos_scores.csv \
  --ppl results/clone/vallex_ppl_mos.csv --divergence results/vallex_grid_pl2_div.csv \
  --group librispeech_pc --out results/clone/master_clone_mos_pl2.csv
# pl2 layer summary (groupby over pl2/seed0 divergence)
python - <<'PY'
import pandas as pd
d=pd.read_csv("results/vallex_grid_pl2_div.csv")
d["config"]="K"+d.key_bits.astype(int).astype(str)+"V"+d.value_bits.astype(int).astype(str)+"@"+d.rw.astype(int).astype(str)
skip={"group","idx","layer","pos","key_bits","value_bits","rw","protected_layers","config"}
m=[c for c in d.columns if c not in skip]
d.groupby(["config","layer"],as_index=False)[m].mean().to_csv("results/vallex_layer_summary_pl2.csv",index=False)
print("wrote results/vallex_layer_summary_pl2.csv")
PY
# rw significance per pl
for pl in pl0 pl2; do
  python tools/rw_analysis.py --scores results/clone/grid_${pl}_mos_scores.csv \
    --out results/clone_rw_significance_${pl}.md --img-dir results/figures_rw_${pl}
done
```

**Laptop** — `scp` the small outputs down, then (torch-free) into `results/final_39speaker/`:
```bash
scp "$REMOTE/results/clone/grid_pl0_mos_scores.csv" "$REMOTE/results/clone/grid_pl2_mos_scores.csv" results/clone/
scp "$REMOTE/results/clone/master_clone_mos_pl2.csv" "$REMOTE/results/vallex_layer_summary_pl2.csv" results/clone/
scp "$REMOTE/results/clone_seed_errorbars_pl0.md" "$REMOTE/results/clone_seed_errorbars_pl2.md" results/final_39speaker/tables/
scp "$REMOTE/results/figures_seed/pl0_cer.png" "$REMOTE/results/figures_seed/pl0_wer.png" \
    "$REMOTE/results/figures_seed/pl2_cer.png" "$REMOTE/results/figures_seed/pl2_wer.png" results/final_39speaker/figures/

D=results/final_39speaker/figures
# pl x rw heatmaps (pl rows, rw cols), all metrics
for m in cer wer spk_sim; do
  .venv/bin/python tools/plot_kv_pl.py \
    --scores "pl0=results/clone/grid_pl0_mos_scores.csv" "pl2=results/clone/grid_pl2_mos_scores.csv" \
    --by-rw --metric $m --out "$D/clone_kv_pl_${m/_sim/sim}_bypl.png"
done
# protection Pareto (correct ratios via =pl)
for m in cer wer; do
  .venv/bin/python tools/plot_tradeoff.py \
    --scores "pl0=results/clone/grid_pl0_mos_scores.csv=0" "pl2=results/clone/grid_pl2_mos_scores.csv=2" \
    --metric $m --out "$D/clone_tradeoff_${m}_bypl.png"
done
```
Then refresh `results/final_39speaker/README.md`: seeds 0/1/2 × pl0/pl2 × rw0/64/128;
remove the single-seed / pl / seed-errorbars caveats.

---
## Verification
- `grid_pl{0,2}_mos_scores.csv`: 28 cfg × seeds {0,1,2} × 195 present.
- `clone_seed_errorbars_pl{0,2}.md`: across-seed std ≪ config gaps; ladder ordering
  identical every seed → **not seed-dependent**.
- pl0 vs pl2 Pareto: protection shifts points but pl0 configs still dominate; by-rw
  panels show rw still inert.
- `results/` holds only 39-speaker artifacts; `results_archive_8spk_*` holds the rest.
