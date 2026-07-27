#!/bin/bash
# A/B test: does the voice-preset LANGUAGE explain VALL-E's high WER?
#
# Runs fp16 VALL-E on the same librispeech_pc sentences with an ENGLISH preset
# vs the Japanese `alan.npz`, scores both (Whisper large-v3 CER/WER), and prints
# the comparison. A large WER drop with the English preset confirms the earlier
# runs were inflated by cross-lingual (ja prompt -> en text) synthesis.
#
# Prereqs on the GPU machine: the repo + env (torch, openai-whisper, jiwer,
# soundfile, librosa, encodec, vocos, transformers), the VALL-E-X checkpoint
# (models/VALL-E-X/checkpoints/vallex-checkpoint.pt), and data/ with
# librispeech_pc (tools/fetch_eval_data.py --fetch-librispeech). Internet for
# the first Whisper large-v3 download.
#
# Usage (from repo root):
#   bash scripts/ab_preset_test.sh [N_SENTENCES] [ENGLISH_PRESET]
#   bash scripts/ab_preset_test.sh 30 librispeech_1.npz
set -euo pipefail

N="${1:-30}"
ENG="${2:-librispeech_1.npz}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

echo "=== A/B preset test: $ENG (en) vs alan.npz (ja), fp16, $N librispeech sentences ==="

for preset in "$ENG" alan.npz; do
  tag="${preset%.npz}"
  echo -e "\n--- generating with preset=$preset (subdir ab_$tag) ---"
  python models/VALL-E-X/benchmarks/benchmark_vallex_real.py --device cuda \
    --groups librispeech_pc --max-per-group "$N" --data-dir data --no-quality \
    --configs fp16 --preset "$preset" --seeds 0 --decode sampling \
    --output-subdir "ab_$tag" --run-tag "ab_$tag"
  echo "--- scoring ab_$tag ---"
  python tools/score_wav_dir.py \
    --audio-dir "models/VALL-E-X/benchmarks/outputs/ab_$tag" \
    --data-dir data --device cuda --out "results/ab_${tag}_scores.csv"
done

echo -e "\n===== A/B RESULT ====="
python3 - "$ENG" <<'PY'
import sys, pandas as pd
eng = sys.argv[1].replace(".npz", "")
for tag, lang in [(eng, "en"), ("alan", "ja")]:
    d = pd.read_csv(f"results/ab_{tag}_scores.csv")
    print(f"{tag:16} ({lang})  n={len(d):3d}  CER {d.cer.mean():.3f}  WER {d.wer.mean():.3f}")
print("\nIf the English preset's WER is much lower, the earlier runs were "
      "inflated by the Japanese cross-lingual prompt -> re-run with the English preset.")
PY
