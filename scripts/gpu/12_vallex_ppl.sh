#!/bin/bash
# Teacher-forced perplexity + attention divergence for VALL-E-X (mechanism
# evidence: per-step damage is tiny, degradation ACCUMULATES in free-running
# generation). EnCodec-tokenizes the ground-truth recordings and runs the AR
# decoder teacher-forced — no sampling, no vocoder, no ASR.
#
#   bash scripts/gpu/12_vallex_ppl.sh
#
# 9 rw=0 configs at pl0 (the fp16 baseline pass is always computed — delta_nll
# and KL are per-sentence relative to it).
# Outputs: results/vallex_ppl_cl.csv (nll/ppl/delta_nll/kl/top1_agree/first_flip)
#          results/vallex_ppl_div_cl.csv (attn_js/cos_k/cos_v/out_cos/relmse)
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate 2>/dev/null || true
mkdir -p results

CFGS="K4V4@0,K4V3@0,K4V2@0,K3V4@0,K3V3@0,K3V2@0,K2V4@0,K2V3@0,K2V2@0"

python models/VALL-E-X/benchmarks/vallex_ppl_divergence.py \
  --groups librispeech_pc --max-per-group "${MAXPG:-100}" --data-dir data \
  --configs "$CFGS" --protected-layers 0 \
  --preset librispeech_1.npz --step-stride 16 \
  --out results/vallex_ppl_cl.csv \
  --divergence-out results/vallex_ppl_div_cl.csv \
  --resume
echo "teacher-forced PPL done"
