#!/bin/bash
# One-time environment setup on a plain GPU machine (Ubuntu/Debian-ish, CUDA 12.x).
# Creates .venv in the repo root and installs everything the pipeline needs.
#
#   bash scripts/gpu/00_setup.sh
#
# Assumes: NVIDIA driver already installed (check with nvidia-smi), python3.10+.
# Needs internet (pip + model downloads happen on first run of each step).
set -euo pipefail
cd "$(dirname "$0")/../.."

# system deps: sox is required by VALL-E-X audio I/O
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -y && sudo apt-get install -y sox libsox-dev ffmpeg
elif command -v brew >/dev/null 2>&1; then
  brew install sox ffmpeg
else
  echo "WARNING: install sox + ffmpeg manually (no apt/brew found)"
fi

# uv (the recommended installer) — bootstrap it if the machine doesn't have it
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# --seed puts pip inside the venv too (the smoke preflight self-heals with pip)
uv venv --python 3.11 --seed .venv
source .venv/bin/activate

# project packages
uv pip install -e .
uv pip install -e models/Qwen3-TTS/
uv pip install -e models/VALL-E-X/

# objective metrics + analysis + tests
uv pip install openai-whisper jiwer soundfile librosa pandas scipy matplotlib pytest

# CUDA torch LAST so nothing downgrades it (matched cu124 build; if your driver
# is older check `nvidia-smi` and switch the index-url to cu121/cu118)
uv pip install --reinstall torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

# tensorflow (preinstalled on some cloud images) breaks WavLM via a cuDNN ABI clash
uv pip uninstall tensorflow tensorflow-cpu tf-keras 2>/dev/null || true

mkdir -p logs results results/figures

python - <<'EOF'
import torch, turboquant
print("ok: torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
EOF
echo "setup done — activate with: source .venv/bin/activate"
