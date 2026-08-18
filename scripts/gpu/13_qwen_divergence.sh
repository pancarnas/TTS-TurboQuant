#!/bin/bash
# Qwen3-TTS counterfactual divergence + wavs, 3 seeds (the cross-model arm).
# Per sentence: generates compressed audio per config AND measures, on one fp16
# pass, how much compression moves the attention maps / KV vectors per layer.
#
#   bash scripts/gpu/13_qwen_divergence.sh
#
# Outputs: results/qwen_div_seed{0,1,2}.csv (per layer/pos/config divergence)
#          models/Qwen3-TTS/benchmarks/outputs/*.wav (scored later by 20_score.sh)
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate 2>/dev/null || true
mkdir -p results logs
unset HF_HUB_OFFLINE 2>/dev/null || true   # tokenizer fetch breaks when offline

CFGS="fp16,K4V4@0,K4V3@0,K3V3@0,K4V2@0,K2V2@0"

for SEED in 0 1 2; do
  echo "### qwen divergence seed $SEED ###"
  python models/Qwen3-TTS/benchmarks/kv_attn_divergence_experiment.py \
    --data-dir data --groups librispeech_pc --max-per-group "${MAXPG:-100}" \
    --configs "$CFGS" --protected-layers 0 --seed "$SEED" \
    --voice-mode clone --device cuda \
    --audio-out-dir models/Qwen3-TTS/benchmarks/outputs \
    --out "results/qwen_div_seed${SEED}.csv" --resume \
    2>&1 | tee "logs/qwen_div_seed${SEED}.log"
done
echo "qwen divergence done"
