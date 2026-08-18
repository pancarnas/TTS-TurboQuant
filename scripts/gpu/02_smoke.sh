#!/bin/bash
# Pre-flight smoke test — run this BEFORE committing the box to the multi-day
# campaign. Three cheap checks (~20-40 min total on one GPU):
#
#   1. unit tests            — KV-cache compression correctness, model-free
#   2. VALL-E-X full smoke   — generate (clone mode, 4 sentences x 5 configs x
#                              2 seeds) -> Whisper/WavLM score -> AUTO-VALIDATE
#                              (cos_k tracks key bits, cos_v value bits, attn_js
#                              ordering, sane fp16 WER). Single PASS/FAIL verdict.
#   3. Qwen3-TTS smoke       — 2 builtin sentences through the quantized
#                              generation path (pipeline check, no metrics)
#
#   bash scripts/gpu/02_smoke.sh
#
# Requires 01_fetch_data.sh first (clone mode needs librispeech_pc references).
# Exits non-zero on any failure — do NOT start 10_vallex_grid.sh until this passes.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate 2>/dev/null || true
mkdir -p logs results

echo "### 1/3 unit tests ###"
python -m pytest models/Qwen3-TTS/tests/ -q

echo "### 2/3 VALL-E-X end-to-end smoke (generate -> score -> validate) ###"
bash scripts/smoke_full.sh 4 librispeech_1.npz clone

echo "### 3/3 Qwen3-TTS generation smoke ###"
python models/Qwen3-TTS/benchmarks/benchmark_qwen3tts_real.py --device cuda \
  --groups smoke --max-per-group 2 --no-quality \
  2>&1 | tee logs/smoke_qwen.log

echo "SMOKE OK — safe to launch the full campaign (10_vallex_grid.sh ...)"
