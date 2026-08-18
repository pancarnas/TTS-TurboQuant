#!/bin/bash
# Full campaign, end to end, on one GPU machine. Each stage is restartable
# individually — if the box dies, rerun the failed stage script, not this file
# (generation regenerates paired-seed wavs deterministically; scoring/PPL/
# divergence use --resume).
#
#   nohup bash scripts/gpu/run_all.sh > logs/run_all.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs

bash scripts/gpu/01_fetch_data.sh
bash scripts/gpu/02_smoke.sh          # aborts the campaign if validation fails
bash scripts/gpu/10_vallex_grid.sh
bash scripts/gpu/11_vallex_seeds.sh
bash scripts/gpu/12_vallex_ppl.sh
bash scripts/gpu/13_qwen_divergence.sh
bash scripts/gpu/20_score.sh
bash scripts/gpu/30_analyze.sh

echo "ALL DONE — tables in results/, figures in results/figures/"
