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

### Quick start (Makefile)

```bash
git clone https://github.com/pancarnas/TTS-TurboQuant.git
cd TTS-TurboQuant
make install-sox      # install sox (audio processing)
make install-cuda     # install torch + torchaudio with CUDA support
make install-all      # install turboquant + Qwen3-TTS + quality metrics
make run              # run Qwen3-TTS benchmark
```

Individual targets:
```bash
make install          # turboquant only
make install-qwen     # turboquant + Qwen3-TTS
make install-metrics  # Whisper CER + WavLM speaker similarity
make install-sox      # auto-detects apt/brew/conda
make install-cuda     # fix CUDA torch if torchaudio errors
```

### Manual setup

#### CUDA (Linux / Cloud GPU)

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
sudo apt-get install -y sox libsox-dev
pip install -e .
pip install -e models/Qwen3-TTS/
pip install openai-whisper jiwer
```

#### macOS (MPS)

```bash
pip install torch torchaudio
brew install sox
pip install -e .
pip install -e models/Qwen3-TTS/
pip install openai-whisper jiwer
```

#### CPU only

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -e .
pip install -e models/Qwen3-TTS/
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
