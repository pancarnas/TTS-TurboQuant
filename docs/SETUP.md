# Alternative setup, Makefile targets & troubleshooting

Everything beyond the recommended uv install in the root
[`README.md`](../README.md): pip/Makefile installs, manual per-platform setup,
quick single-model checks, and troubleshooting.

## Install with pip / Makefile

```bash
git clone https://github.com/pancarnas/TTS-TurboQuant.git
cd TTS-TurboQuant
make install-cuda install-sox install-all install-vallex
```

Four chained Makefile targets in a single `make` invocation:

| Target | Installs |
|---|---|
| `install-cuda` | torch + torchaudio on CUDA 12.4 |
| `install-sox` | `libsox` system lib (auto-detects apt / brew / conda) |
| `install-all` | turboquant + Qwen3-TTS deps + Whisper + jiwer (quality metrics) |
| `install-vallex` | VALL-E-X deps (encodec, vocos, tokenizers, jieba, …) |

## Individual install targets (if you only want part of the stack)

```bash
make install          # turboquant only
make install-qwen     # turboquant + Qwen3-TTS
make install-vallex   # turboquant + VALL-E-X
make install-all      # turboquant + Qwen3-TTS + quality metrics
make install-metrics  # Whisper CER + WavLM speaker similarity
make install-sox      # auto-detects apt/brew/conda
make install-cuda     # fix CUDA torch if torchaudio errors
```

## Manual setup (if not using the Makefile)

### CUDA (Linux / Cloud GPU)

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
sudo apt-get install -y sox libsox-dev
pip install -e .                          # turboquant
pip install -e models/Qwen3-TTS/          # Qwen3-TTS
pip install -e models/VALL-E-X/           # VALL-E-X
pip install openai-whisper jiwer pytest   # quality metrics + tests
# Optional: remove tensorflow to fix WavLM on SageMaker
pip uninstall -y tensorflow tensorflow-cpu tf-keras 2>/dev/null; true
```

### macOS (MPS)

```bash
pip install torch torchaudio
brew install sox
pip install -e .
pip install -e models/Qwen3-TTS/
pip install -e models/VALL-E-X/
pip install openai-whisper jiwer pytest
```

### CPU only

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -e .
pip install -e models/Qwen3-TTS/
pip install -e models/VALL-E-X/
```

## Quick single-model checks (Makefile)

```bash
make run              # Qwen3-TTS 22-sentence sweep
make run DEVICE=mps   # specify device
make run-vallex       # VALL-E-X 22-sentence sweep
make smoke-vallex     # VALL-E-X quick pipeline check (2 short sentences)
make smoke-qwen       # Qwen3-TTS quick pipeline check
make profile-vallex   # torch.profiler on VALL-E-X (Perfetto traces)
make profile-qwen     # torch.profiler on Qwen3-TTS
make evaluate         # quality metrics on saved wavs (no TTS model needed)
make test             # unit tests
```

Running on the Eddie cluster (Grid Engine + shared conda env): see
[`EDDIE.md`](EDDIE.md).

## Troubleshooting

### `libcudart.so` / torchaudio CUDA errors
Your `torchaudio` was built for a different CUDA version. Reinstall matching your system:
```bash
nvidia-smi | head -3  # check CUDA version
pip install torch torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/cu124
```

### `torchcodec` errors from `torchaudio.load`
(`ModuleNotFoundError: No module named 'torchcodec'` or `TorchCodec is
required for load_with_torchcodec`.) torchaudio ≥ 2.8 moved audio decoding
into the separate `torchcodec` package, which additionally needs FFmpeg
shared libraries. The reliable fix is the tested stack:
```bash
pip install --force-reinstall torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124
```
To keep a newer torch instead, install torchcodec without letting pip upgrade
torch, plus FFmpeg:
```bash
pip install torchcodec "torch==$(python -c 'import torch; print(torch.__version__.split("+")[0])')"
conda install -y -c conda-forge 'ffmpeg<8'   # or: sudo apt-get install -y ffmpeg
```
`scripts/gpu/02_smoke.sh` probes a real decode in its preflight and applies
the torchcodec route automatically before failing with these instructions.

### `sox: not found`
```bash
sudo apt-get install -y sox libsox-dev   # Ubuntu/Debian
conda install -y -c conda-forge sox      # Conda
brew install sox                          # macOS
```
