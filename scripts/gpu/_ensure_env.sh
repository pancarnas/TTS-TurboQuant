#!/bin/bash
# Sourced by the GPU scripts: pin the .venv interpreter and self-heal the venv.
# A box restart (or a stray `uv sync`, which strips every package pyproject does
# not list) leaves .venv missing pandas/soundfile/whisper/qwen_tts... Instead of
# dying one missing module at a time, install the full tool set once (uv pip is
# idempotent: already-present packages are skipped in a second or two).
PY="${PYTHON:-}"
[ -z "$PY" ] && { [ -x .venv/bin/python ] && PY=.venv/bin/python || PY=python; }
_need='import pandas, scipy, soundfile, librosa, whisper, jiwer, tqdm, transformers, torch, torchaudio, turboquant, qwen_tts'
if ! "$PY" -c "$_need" 2>/dev/null; then
  echo "### venv incomplete — installing the tool set into $PY ###"
  UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
  [ -x "$UV" ] || { curl -LsSf https://astral.sh/uv/install.sh | sh; UV="$HOME/.local/bin/uv"; }
  "$UV" pip install --python "$PY" -e . -e models/Qwen3-TTS/ \
    openai-whisper jiwer soundfile librosa pandas scipy matplotlib pytest tqdm
  "$PY" -c "$_need" || { echo "venv still broken after install — run: bash scripts/gpu/00_setup.sh" >&2; exit 2; }
fi
"$PY" -c 'import torch, sys; ok = torch.cuda.is_available(); print("torch", torch.__version__, "cuda", ok); sys.exit(0 if ok else 3)' \
  || { echo "torch has no CUDA — run: bash scripts/gpu/00_setup.sh (reinstalls torch 2.6.0+cu124)" >&2; exit 3; }
