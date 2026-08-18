# TTS-TurboQuant

Integration of [TurboQuant](https://arxiv.org/pdf/2504.19874v1) KV cache compression into [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS). Reduces KV cache memory by 2-4x during autoregressive speech generation with minimal quality loss.

Based on the [TurboQuant reference implementation](https://github.com/0xSero/turboquant).

## Results

### Main campaign — VALL-E-X, zero-shot voice cloning (LibriSpeech-PC)

39 held-out speakers × 5 sentences = 195 sentences, per-item reference voice
(`--voice-mode clone`), seeds 0/1/2, no protected layers (pl0). Objective
metrics: Whisper large-v3 CER/WER vs ground-truth text, WavLM speaker cosine
vs the same sentence's fp16 output. Mean ± std across seeds; full 28-config
grid (K,V ∈ {4,3,2} × residual window {0,64,128}) in the thesis tables.

| Config | KV bits | CER | WER | SpkSim |
|--------|---------|-----|-----|--------|
| fp16 baseline | 16 | 0.056 ± 0.005 | 0.104 ± 0.010 | — |
| K3V4@0 | 3.5 | 0.061 ± 0.008 | 0.107 ± 0.011 | 0.954 |
| K4V4@0 | 4 | 0.063 ± 0.008 | 0.114 ± 0.010 | 0.955 |
| **K3V3@0** | **3** | **0.067 ± 0.003** | **0.118 ± 0.004** | **0.951** |
| K4V2@0 | 3 | 0.084 ± 0.006 | 0.147 ± 0.014 | 0.948 |
| K2V2@0 | 2 | 0.127 ± 0.004 | 0.209 ± 0.007 | 0.946 |

Key findings:

- **3-bit KV is free** — K4V4/K3V3/K3V4 overlap the fp16 baseline within ~1
  seed-std; the floor is **2-bit keys** (K2V2 ≈ 2× baseline WER, but degrades
  gracefully — no collapse).
- **Cross-model contrast** — at matched bits and near-identical KV
  reconstruction (cos_k: K4 .996 / K3 .984 / K2 .942 on both models), Qwen3-TTS
  (GQA) collapses on 54–100% of sentences while VALL-E-X (MHA) collapses on ~0%.
  The mechanism is the attention response (Qwen layer-5 attn_js hotspot vs
  VALL-E flat), not reconstruction error.
- **Damage accumulates** — teacher-forced perplexity barely moves at K2V2 while
  free-running WER doubles: per-step error is tiny, the autoregressive loop
  compounds it.
- **Timbre survives** — speaker similarity stays ~0.95 across all configs, even
  where CER breaks (right voice, wrong words).

Reproduce end to end with [`scripts/gpu/README.md`](scripts/gpu/README.md).

### Earlier integration benchmark — Qwen3-TTS 1.7B custom-voice (22 sentences)

10 short / 7 medium / 5 long sentences, preset voice, single seed.

| Config | Bits | CER (short) | CER (med) | CER (long) | Speaker Sim | Attn Sim | Speed (long) |
|--------|------|-------------|-----------|------------|-------------|----------|--------------|
| Baseline | 16 | 0.65% | 1.05% | 3.17% | --- | --- | 1.0x |
| K4/V2 | 3 | 0.29% | 0.56% | 3.60% | 0.99 | 0.97 | 1.4x |
| **K3/V3** | **3** | **0.65%** | **0.37%** | **5.28%** | **0.99** | **0.99** | **1.4x** |
| K3/V2 | 2.5 | 0.00% | 0.77% | 4.20% | 0.99 | 0.97 | 1.4x |
| K2/V2 | 2 | 0.00% | 0.81% | 72.1% | 0.97 | 0.95 | 1.5x |

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
uv pip install openai-whisper jiwer soundfile librosa pandas scipy matplotlib pytest

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

### Run the experiment pipeline

Full campaign on a single GPU machine, from raw data to the thesis tables and
figures. Run the stages in order (each is restartable on its own):

```bash
source .venv/bin/activate

# 1. evaluation data: seed-tts-eval + ellav_hard + LibriSpeech-PC (~400 MB)
bash scripts/gpu/01_fetch_data.sh

# 2. pre-flight smoke — unit tests + tiny generate→score→validate round trip.
#    MUST print PASS before launching anything heavy (~20-40 min).
bash scripts/gpu/02_smoke.sh

# 3. generation
bash scripts/gpu/10_vallex_grid.sh       # VALL-E-X 28-config K×V×rw grid, pl0+pl2
bash scripts/gpu/11_vallex_seeds.sh      # 5 headline configs × seeds 0/1/2
bash scripts/gpu/12_vallex_ppl.sh        # teacher-forced PPL/KL (mechanism)
bash scripts/gpu/13_qwen_divergence.sh   # Qwen cross-model arm, 3 seeds

# 4. objective metrics: Whisper large-v3 CER/WER + WavLM speaker similarity
bash scripts/gpu/20_score.sh

# 5. tables, statistics and figures (CPU-only, also runs on a laptop)
bash scripts/gpu/30_analyze.sh
```

Or run everything unattended (aborts automatically if the smoke fails):

```bash
nohup bash scripts/gpu/run_all.sh > logs/run_all.log 2>&1 &
```

Knobs (`NSHARDS`, `MAXPG`), per-stage runtimes, resume behaviour and
troubleshooting: [`scripts/gpu/README.md`](scripts/gpu/README.md).

### Quick single-model checks (Makefile)

```bash
make run              # Qwen3-TTS 22-sentence sweep
make run-vallex       # VALL-E-X 22-sentence sweep
make smoke-vallex     # VALL-E-X quick pipeline check (2 short sentences)
make smoke-qwen       # Qwen3-TTS quick pipeline check
make profile-vallex   # torch.profiler on VALL-E-X (Perfetto traces)
make profile-qwen     # torch.profiler on Qwen3-TTS
```

Running on the Eddie cluster (Grid Engine + shared conda env): see
[`docs/EDDIE.md`](docs/EDDIE.md).

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
pip install openai-whisper jiwer pytest   # quality metrics + tests
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
pip install openai-whisper jiwer pytest
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

For the full experiment campaign (smoke → generation → objective metrics →
analysis) see [Run the experiment pipeline](#run-the-experiment-pipeline)
above and [`scripts/gpu/README.md`](scripts/gpu/README.md).

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
