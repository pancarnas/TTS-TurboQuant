#!/usr/bin/env bash
# Key/value Gaussianity + alignment diagnostics on Qwen3-TTS (OmniKVQuant Table 1
# analogue). One fp16 greedy pass per sentence, no quantization on the path.
#   MAXPG=20 bash scripts/gpu/15_qwen_keygauss.sh
# Outputs: results/keygauss_stats.csv, results/keygauss_stats_layers.csv,
#          results/keygauss_stats_summary.md
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate 2>/dev/null || true
source scripts/gpu/_ensure_env.sh
mkdir -p results logs
unset HF_HUB_OFFLINE 2>/dev/null || true
MAXPG="${MAXPG:-20}"
EVAL_GROUPS="${EVAL_GROUPS:-librispeech_pc}"  # not GROUPS: that is a bash builtin (unix gids)
BITS="${BITS:-2,3}"
"$PY" models/Qwen3-TTS/benchmarks/key_gauss_stats.py \
  --data-dir data --groups "$EVAL_GROUPS" --max-per-group "$MAXPG" --bits "$BITS" \
  --voice-mode clone --device cuda --out results/keygauss_stats.csv \
  2>&1 | tee logs/qwen_keygauss.log
echo "keygauss done - see results/keygauss_stats_summary.md"
