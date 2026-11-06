#!/bin/bash
# Pre-flight smoke for the KEY-FIX experiments (docs/keyfix_experiment.md).
# Run this BEFORE 14_qwen_keyfix.sh. ~10-25 min on one GPU:
#
#   1. unit tests        — compressor group scaling, RoPE undo/redo, both
#                          caches' span rounding, recorder, parsers (CPU)
#   2. three tiny cells  — 2 librispeech_pc sentences x {fp16, K8V8@0, K2V2@0}
#                          through the Qwen divergence experiment, once per
#                          cell: plain / --key-group 32 / --pre-rope-keys
#   3. AUTO-VALIDATE     — fp16 wavs identical across cells (flags never touch
#                          the baseline), K2V2 wavs differ across cells (flags
#                          reach generation), K8V8 pre-RoPE cos_k ~ 1 (RoPE is
#                          undone exactly on the real mRoPE tables), CSV
#                          columns carry the cell flags. Single PASS/FAIL.
#
#   bash scripts/gpu/03_smoke_keyfix.sh
#   CELLS="pre" bash scripts/gpu/03_smoke_keyfix.sh   # re-run ONE cell after a fix;
#                                                     # the other cells' CSVs/wavs are kept
#
# Requires 01_fetch_data.sh (librispeech_pc references for clone mode).
# Exits non-zero on any failure — do NOT start 14_qwen_keyfix.sh until it passes.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate 2>/dev/null || true
mkdir -p logs results
unset HF_HUB_OFFLINE 2>/dev/null || true

OUT_DIR=models/Qwen3-TTS/benchmarks/outputs/smoke_keyfix
CFGS="fp16,K8V8@0,K2V2@0"
ALL_CELLS="plain g32 pre"
CELLS="${CELLS:-$ALL_CELLS}"

echo "### 1/3 unit tests (key-fix + divergence driver + caches) ###"
python -c "import pytest" 2>/dev/null || pip install -q pytest
python -m pytest tests/test_keyfix.py tests/test_kv_attn_divergence.py \
  tests/test_turboquant_cache_readback.py tests/test_score_wav_dir.py \
  tests/test_check_audio.py models/Qwen3-TTS/tests/ -q

echo "### 2/3 three tiny cells through the Qwen divergence experiment ###"
[ "$CELLS" = "$ALL_CELLS" ] && rm -rf "$OUT_DIR"
for CELL in $CELLS; do
  case "$CELL" in
    plain) FLAGS="" ;;
    g32)   FLAGS="--key-group 32" ;;
    pre)   FLAGS="--pre-rope-keys" ;;
  esac
  echo "--- cell $CELL ($FLAGS) ---"
  # shellcheck disable=SC2086
  python models/Qwen3-TTS/benchmarks/kv_attn_divergence_experiment.py \
    --data-dir data --groups librispeech_pc --max-per-group 2 \
    --configs "$CFGS" --protected-layers 0 --seed 0 \
    --voice-mode clone --device cuda \
    --audio-out-dir "$OUT_DIR" \
    --out "results/smoke_keyfix_${CELL}.csv" $FLAGS \
    2>&1 | tee "logs/smoke_keyfix_${CELL}.log"
done

echo "### 3/3 validate (all three cells, whichever were re-run) ###"
python tools/validate_keyfix_smoke.py --results results --wavdir "$OUT_DIR" --n 2 --logs logs
