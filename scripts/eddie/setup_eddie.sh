#!/bin/bash
# One-time Eddie setup for TTS-TurboQuant.
#
# Run this ONCE on a login node (login nodes have internet; compute nodes are for
# the actual GPU jobs), from the repo root:
#   bash scripts/eddie/setup_eddie.sh
#
# It uses the shared, read-only conda env `py312torch27cuda118` for the GPU torch
# build and installs ONLY the third-party deps that env is missing into a repo-local
# user-site. It never reinstalls torch/torchaudio. The repo's own pure-Python
# packages stay on PYTHONPATH (see eddie_run.sh), so nothing is installed into the
# shared env.

set -euo pipefail

. /etc/profile.d/modules.sh
module load anaconda
conda activate py312torch27cuda118

export PYTHONUSERBASE="$PWD/.eddie-userbase"
mkdir -p logs "$PYTHONUSERBASE" .hf_cache

echo "Installing third-party deps into user-site: $PYTHONUSERBASE"
python -m pip install --user --upgrade pip

# torch / torchaudio are intentionally NOT listed — the shared env provides the
# GPU build. If any package below tries to drag in a different torch, re-run just
# that package with `--no-deps`.
python -m pip install --user \
    "transformers==4.57.3" "accelerate==1.12.0" \
    librosa soundfile onnxruntime einops "scipy>=1.10.0" \
    openai-whisper jiwer pandas \
    encodec vocos tokenizers unidecode inflect eng_to_ipa jieba cn2an wget tqdm

echo
echo "Verifying torch + repo imports (using the same PYTHONPATH the jobs use)..."
PYTHONPATH="$PWD:$PWD/models/Qwen3-TTS:$PWD/models/VALL-E-X:${PYTHONPATH:-}" \
    python -c "import torch, turboquant; print('ok: torch', torch.__version__)"

echo
echo "Setup complete."
echo "Before submitting jobs, edit the '#\$ -M' email line in scripts/eddie/eddie_run.sh."
echo "Then submit, e.g.:"
echo "  qsub -N tq_smoke -l h_rt=0:30:00 scripts/eddie/eddie_run.sh make smoke-qwen DEVICE=cuda"
