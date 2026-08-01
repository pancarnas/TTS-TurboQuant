#!/bin/bash
# Score the CLONE re-run wavs (Whisper large-v3 CER/WER + WavLM speaker sim)
# into results/clone/. Run on a GPU node via qsub + eddie_run.sh.
#
#   qsub -N vx_score_clone -l h_rt=12:00:00 -pe sharedmem 4 -l gpu=1 \
#     scripts/eddie/eddie_run.sh bash scripts/eddie/score_clone.sh
#
# --resume makes it restartable (skips already-scored rows) if it times out.
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p results/clone
for sub in grid_clone_pl0 grid_clone_pl2 seed_clone_pl0 seed_clone_pl2; do
  echo "### scoring $sub ###"
  python tools/score_wav_dir.py \
    --audio-dir "models/VALL-E-X/benchmarks/outputs/$sub" \
    --data-dir data --device cuda \
    --out "results/clone/${sub}_scores.csv" --resume
done
echo "all clone scoring done"
