#!/bin/bash
# VALL-E-X multi-seed headline run (error bars): 5 configs x seeds 0,1,2,
# clone protocol, pl0 and pl2. Each seed is a paired draw per config, so
# config-to-config deltas are per-sentence paired.
#
#   bash scripts/gpu/11_vallex_seeds.sh
#
# Outputs: models/VALL-E-X/benchmarks/outputs/seed_cl_pl{0,2}/*.wav
#          results/seed_cl_pl{0,2}_div_shard*.csv
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate 2>/dev/null || true
mkdir -p logs results

NSHARDS="${NSHARDS:-2}"
MAXPG="${MAXPG:-100}"
CFGS="fp16,K4V4@0,K4V2@0,K3V3@0,K2V2@0"

for PL in 0 2; do
  TAG="seed_cl_pl${PL}"
  echo "### $TAG : 5 configs x 3 seeds ###"
  bash scripts/eddie/run_vallex_perm.sh "$NSHARDS" "$MAXPG" librispeech_pc \
    "$TAG" "$CFGS" "$PL" "$TAG" 0,1,2 librispeech_1.npz div clone
done
echo "multi-seed generation done"
