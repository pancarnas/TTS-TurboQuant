#!/bin/bash
# Download the evaluation data into data/ (needs internet, ~400 MB).
#   - seed-tts-eval test-en   (HuggingFace)
#   - ellav_hard              (text-only reconstruction, written locally)
#   - LibriSpeech-PC          (F5-TTS cross-sentence .lst + OpenSLR test-clean)
#
# librispeech_pc is the group used for the clone-mode campaign — it is the only
# set with a per-item reference voice AND a ground-truth recording per sentence.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate 2>/dev/null || true

python tools/fetch_eval_data.py --data-dir data --fetch-librispeech
python tools/validate_eval_set.py --data-dir data
echo "data ready under data/"
