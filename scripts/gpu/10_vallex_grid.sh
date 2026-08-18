#!/bin/bash
# VALL-E-X full bit x residual-window grid, zero-shot CLONE protocol, seed 0.
# 27 quantized configs (K,V in {4,3,2} x rw in {0,64,128}) + fp16 baseline,
# run twice: protected_layers=0 (deployment point) and protected_layers=2.
# Attention divergence is recorded in the same pass (one generation -> wavs +
# per-(config,layer,pos) divergence CSV shards).
#
#   bash scripts/gpu/10_vallex_grid.sh
#
# Knobs (env): NSHARDS=2 parallel workers on the one GPU (raise to 3-4 on
# 40GB+ cards, drop to 1 if you OOM), MAXPG=100 sentences from librispeech_pc.
# Outputs: models/VALL-E-X/benchmarks/outputs/grid_cl_pl{0,2}/*.wav
#          results/grid_cl_pl{0,2}_div_shard*.csv, logs/grid_cl_pl{0,2}_shard*.log
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate 2>/dev/null || true
mkdir -p logs results

NSHARDS="${NSHARDS:-2}"
MAXPG="${MAXPG:-100}"

GRID="fp16"
for K in 4 3 2; do for V in 4 3 2; do for RW in 0 64 128; do
  GRID="$GRID,K${K}V${V}@${RW}"
done; done; done

for PL in 0 2; do
  TAG="grid_cl_pl${PL}"
  echo "### $TAG : $NSHARDS shards, $MAXPG sentences, 28 configs ###"
  bash scripts/eddie/run_vallex_perm.sh "$NSHARDS" "$MAXPG" librispeech_pc \
    "$TAG" "$GRID" "$PL" "$TAG" 0 librispeech_1.npz div clone
done
echo "grid generation done"
