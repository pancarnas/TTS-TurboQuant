# TTS-TurboQuant

Integration of [TurboQuant](https://arxiv.org/abs/2406.02525) KV cache compression into [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS). Reduces KV cache memory by 2-4x during autoregressive speech generation with minimal quality loss.

Based on the [TurboQuant reference implementation](https://github.com/0xSero/turboquant).

## Results

### Real-weights benchmark (Qwen3-TTS 1.7B, bfloat16, CUDA)

22 sentences: 10 short (~7 words), 7 medium (~24 words), 5 long (~76 words). Quality measured with Whisper CER and WavLM speaker cosine similarity.

**Averaged results:**

| Config | Bits | CER (short) | CER (med) | CER (long) | Speaker Sim | Attn Sim | Speed (long) |
|--------|------|-------------|-----------|------------|-------------|----------|--------------|
| Baseline | 16 | 0.65% | 1.05% | 3.17% | --- | --- | 1.0x |
| K4/V2 | 3 | 0.29% | 0.56% | 3.60% | 0.99 | 0.97 | 1.4x |
| **K3/V3** | **3** | **0.65%** | **0.37%** | **5.28%** | **0.99** | **0.99** | **1.4x** |
| K3/V2 | 2.5 | 0.00% | 0.77% | 4.20% | 0.99 | 0.97 | 1.4x |
| K2/V2 | 2 | 0.00% | 0.81% | 72.1% | 0.97 | 0.95 | 1.5x |

**Attention similarity (KV reconstruction cosine similarity):**

| Config | Attn Sim |
|--------|----------|
| **K3/V3** | **0.9855** |
| K4/V2 | 0.9726 |
| K3/V2 | 0.9674 |
| K2/V2 | 0.9492 |

Note: attention similarity is input-agnostic by design — random rotation makes the per-coordinate distribution uniform regardless of sentence content.

## Project structure

```
TTS-TurboQuant/
├── turboquant/              # Shared compression library
│   ├── config.py            # TurboQuantConfig
│   ├── compressors_v3.py    # TurboQuantV3 — production compressor
│   ├── lloyd_max.py         # Lloyd-Max optimal scalar quantizer
│   └── pyproject.toml       # Installable as: pip install -e turboquant/
├── models/
│   └── Qwen3-TTS/           # Qwen3-TTS with TurboQuant integration
│       ├── qwen_tts/        # Core package
│       ├── benchmarks/      # Real-weights benchmarks
│       ├── tests/           # Unit tests for KV cache compression
│       └── pyproject.toml   # Installable as: pip install -e models/Qwen3-TTS/
├── examples/
│   └── qwen3_tts_turboquant.py  # Qwen3-TTS with TurboQuant
├── Makefile                 # make install, make run, make test
└── pyproject.toml           # Root package (installs turboquant)
```

## Setup

### Install with uv (recommended)

```bash
git clone https://github.com/pancarnas/TTS-TurboQuant.git
cd TTS-TurboQuant

uv venv --python 3.11 .venv
source .venv/bin/activate

# project packages + quality metrics
uv pip install -e .
uv pip install -e models/Qwen3-TTS/
uv pip install -e models/VALL-E-X/
uv pip install openai-whisper jiwer soundfile librosa pandas scipy

# CUDA torch installed last (matched cu124 build)
uv pip install --reinstall torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124
```

### Quick start with pip / Makefile (alternative)

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

On SageMaker, you may additionally need to remove tensorflow to avoid a cuDNN ABI conflict that breaks WavLM speaker similarity:
```bash
pip uninstall -y tensorflow tensorflow-cpu tf-keras 2>/dev/null; true
```

### Then run a benchmark

```bash
make run              # Qwen3-TTS 22-sentence sweep
make run-vallex       # VALL-E-X 22-sentence sweep
make smoke-vallex     # VALL-E-X quick pipeline check (2 short sentences)
make smoke-qwen       # Qwen3-TTS quick pipeline check
make profile-vallex   # torch.profiler on VALL-E-X (Perfetto traces)
make profile-qwen     # torch.profiler on Qwen3-TTS
```

### Individual install targets (if you only want part of the stack)

```bash
make install          # turboquant only
make install-qwen     # turboquant + Qwen3-TTS
make install-vallex   # turboquant + VALL-E-X
make install-all      # turboquant + Qwen3-TTS + quality metrics
make install-metrics  # Whisper CER + WavLM speaker similarity
make install-sox      # auto-detects apt/brew/conda
make install-cuda     # fix CUDA torch if torchaudio errors
```

### Manual setup (if not using the Makefile)

#### CUDA (Linux / Cloud GPU)

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
sudo apt-get install -y sox libsox-dev
pip install -e .                          # turboquant
pip install -e models/Qwen3-TTS/          # Qwen3-TTS
pip install -e models/VALL-E-X/           # VALL-E-X
pip install openai-whisper jiwer          # quality metrics
# Optional: remove tensorflow to fix WavLM on SageMaker
pip uninstall -y tensorflow tensorflow-cpu tf-keras 2>/dev/null; true
```

#### macOS (MPS)

```bash
pip install torch torchaudio
brew install sox
pip install -e .
pip install -e models/Qwen3-TTS/
pip install -e models/VALL-E-X/
pip install openai-whisper jiwer
```

#### CPU only

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -e .
pip install -e models/Qwen3-TTS/
pip install -e models/VALL-E-X/
```

## Quick start

```python
import torch
from qwen_tts import Qwen3TTSModel
from turboquant import TurboQuantConfig

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device_map="cuda",  # or "mps" for Mac, "cpu" for CPU
    dtype=torch.bfloat16,
)

# Generate with KV cache compression
wavs, sr = model.generate_custom_voice(
    text="Hello, how are you?",
    language="English",
    speaker="Ryan",
    turboquant_config=TurboQuantConfig(key_bits=4, value_bits=2),
)

# Inspect compression stats
cache = model.model.last_kv_cache
print(cache.memory_report())
```

## Benchmarks

```bash
make run                  # run benchmark with quality metrics
make run DEVICE=mps       # specify device

# Evaluate quality on saved wavs (no TTS model needed)
make evaluate
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `key_bits` | 4 | Bits per key coordinate (higher = better quality) |
| `value_bits` | 2 | Bits per value coordinate |
| `residual_window` | 128 | Recent tokens kept in fp16 |
| `protected_layers` | 2 | First/last N layers use higher precision |
| `protected_bits` | 8 | Bit-width for protected layers |
| `enabled` | True | Global on/off switch |

## Testing

```bash
make test
```

## Troubleshooting

### `libcudart.so` / torchaudio CUDA errors
Your `torchaudio` was built for a different CUDA version. Reinstall matching your system:
```bash
nvidia-smi | head -3  # check CUDA version
pip install torch torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/cu124
```

### `sox: not found`
```bash
sudo apt-get install -y sox libsox-dev   # Ubuntu/Debian
conda install -y -c conda-forge sox      # Conda
brew install sox                          # macOS
```

## License

- TurboQuant compression library: MIT
- Qwen3-TTS model code: Apache 2.0 (Alibaba Qwen Team)
