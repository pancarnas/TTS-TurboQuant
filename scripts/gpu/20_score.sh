#!/bin/bash
# Objective-metric scoring of every generated wav (GPU strongly recommended):
#   CER + WER  — Whisper large-v3 transcription vs ground-truth text
#                (EnglishTextNormalizer applied to both ref and hyp)
#   spk_sim    — WavLM x-vector cosine of each compressed wav against the SAME
#                sentence's fp16 baseline wav (same arm/seed/temperature)
#
#   bash scripts/gpu/20_score.sh
#
# --resume makes every pass restartable (skips already-scored rows), so you can
# also run this mid-campaign and re-run it at the end.
# Outputs: results/{grid,seed}_cl_pl{0,2}_scores.csv, results/qwen_seed_scores.csv
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate 2>/dev/null || true
mkdir -p results

for SUB in grid_cl_pl0 grid_cl_pl2 seed_cl_pl0 seed_cl_pl2; do
  DIR="models/VALL-E-X/benchmarks/outputs/$SUB"
  if [ -d "$DIR" ]; then
    echo "### scoring $SUB ###"
    python tools/score_wav_dir.py --audio-dir "$DIR" --data-dir data \
      --device cuda --out "results/${SUB}_scores.csv" --resume
  else
    echo "skip $SUB (no wavs at $DIR)"
  fi
done

if [ -d models/Qwen3-TTS/benchmarks/outputs ]; then
  echo "### scoring qwen ###"
  python tools/score_wav_dir.py --audio-dir models/Qwen3-TTS/benchmarks/outputs \
    --data-dir data --device cuda --out results/qwen_seed_scores.csv --resume
fi
echo "scoring done"
