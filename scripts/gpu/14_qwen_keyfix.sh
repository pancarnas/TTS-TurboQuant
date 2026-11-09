#!/bin/bash
# KEY-FIX experiment on Qwen3-TTS (docs/keyfix_experiment.md): does the K2
# collapse go away with (E1) OmniKVQuant-style per-group key scaling and/or
# (E2) pre-RoPE key quantization? One invocation of the divergence experiment
# per CELL; every cell generates the compressed wavs AND records per-layer
# attention divergence on an fp16 pass, exactly like 13_qwen_divergence.sh.
#
#   bash scripts/gpu/14_qwen_keyfix.sh
#
# Knobs (env):
#   CELLS="g32 pre g32pre"   cells to run. 'plain' (no flags) is the baseline
#                            already produced by 13_qwen_divergence.sh seed 0
#                            for K2V2@0/K3V3@0; add it here to regenerate
#                            under this CFGS list (e.g. for K2V4@0).
#   CFGS="fp16,K2V2@0,K2V4@0,K3V3@0"   configs per cell (fp16 = audio baseline)
#   SEEDS="0"                paired seeds (3 = error bars, 3x the time)
#   MAXPG=100                librispeech_pc sentences (50 halves everything)
#   SCORE=1                  run Whisper/WavLM scoring per cell afterwards
#
# Rough cost per cell on an A100-40GB at MAXPG=100: 100 sentences x
# (3 TQ configs + fp16 + fp16 divergence pass) = 500 generations ~ 3-6 h.
# Default 3 cells, 1 seed: 9-18 h. Everything is --resume-able per sentence.
#
# Outputs: results/qwen_keyfix_<cell>_seed<s>.csv       (divergence rows)
#          models/Qwen3-TTS/benchmarks/outputs/keyfix_<cell>/*.wav
#          results/qwen_keyfix_<cell>_scores.csv        (if SCORE=1)
#          results/keyfix_summary.md                    (tools/summarize_keyfix.py)
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate 2>/dev/null || true
# pin the interpreter: a fresh shell after a box restart has no venv on PATH
PY="${PYTHON:-}"; [ -z "$PY" ] && { [ -x .venv/bin/python ] && PY=.venv/bin/python || PY=python; }
"$PY" -c "import pandas, torch" || { echo "venv broken: $PY lacks pandas/torch — run scripts/gpu/00_setup.sh" >&2; exit 2; }
mkdir -p results logs
unset HF_HUB_OFFLINE 2>/dev/null || true

CELLS="${CELLS:-g32 pre g32pre}"
CFGS="${CFGS:-fp16,K2V2@0,K2V4@0,K3V3@0}"
SEEDS="${SEEDS:-0}"
MAXPG="${MAXPG:-100}"
SCORE="${SCORE:-1}"

cell_flags() {
  case "$1" in
    plain)  echo "" ;;
    g32)    echo "--key-group 32" ;;
    pre)    echo "--pre-rope-keys" ;;
    g32pre) echo "--key-group 32 --pre-rope-keys" ;;
    *) echo "unknown cell '$1' (plain|g32|pre|g32pre)" >&2; exit 2 ;;
  esac
}

for CELL in $CELLS; do
  FLAGS="$(cell_flags "$CELL")"
  OUT_DIR="models/Qwen3-TTS/benchmarks/outputs/keyfix_${CELL}"
  for SEED in $SEEDS; do
    echo "### keyfix cell=$CELL seed=$SEED flags='$FLAGS' ###"
    # shellcheck disable=SC2086
    "$PY" models/Qwen3-TTS/benchmarks/kv_attn_divergence_experiment.py \
      --data-dir data --groups librispeech_pc --max-per-group "$MAXPG" \
      --configs "$CFGS" --protected-layers 0 --seed "$SEED" \
      --voice-mode clone --device cuda \
      --audio-out-dir "$OUT_DIR" \
      --out "results/qwen_keyfix_${CELL}_seed${SEED}.csv" --resume $FLAGS \
      2>&1 | tee "logs/qwen_keyfix_${CELL}_seed${SEED}.log"
    # record() swallows per-step errors so one bad step cannot kill a 5-hour
    # run; but a cell with skipped measurements is NOT a result. Stop here.
    if grep -q "measurement errors" "logs/qwen_keyfix_${CELL}_seed${SEED}.log"; then
      echo "ABORT: cell=$CELL seed=$SEED had measurement errors — see the log above" >&2
      exit 3
    fi
  done
  if [ "$SCORE" = "1" ]; then
    echo "### scoring cell=$CELL (Whisper large-v3 CER/WER + WavLM spk_sim vs fp16) ###"
    "$PY" tools/score_wav_dir.py --audio-dir "$OUT_DIR" --data-dir data \
      --device cuda --out "results/qwen_keyfix_${CELL}_scores.csv" --resume \
      2>&1 | tee "logs/qwen_keyfix_${CELL}_score.log"
  fi
done

echo "### summary ###"
# 'plain' baseline: this run's own cell if CELLS had one, else the seed-0
# outputs of 13_qwen_divergence.sh when they are on this box (restricted to the
# same groups/seeds by the summariser).
PLAIN=""
if [[ " $CELLS " != *" plain "* ]]; then
  [ -f results/qwen_div_seed0.csv ] && PLAIN="$PLAIN --plain-div results/qwen_div_seed0.csv"
  [ -f results/qwen_seed_scores.csv ] && PLAIN="$PLAIN --plain-scores results/qwen_seed_scores.csv"
fi
# shellcheck disable=SC2086
"$PY" tools/summarize_keyfix.py --results results --out results/keyfix_summary.md $PLAIN || true
echo "keyfix done — see results/keyfix_summary.md"
