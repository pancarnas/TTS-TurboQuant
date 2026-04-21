.PHONY: install install-qwen install-vallex install-vallex-all install-all install-metrics install-sox install-cuda run run-vallex smoke-vallex profile-vallex profile-vallex-full evaluate evaluate-vallex test clean

DEVICE ?= $(shell python -c "import torch; print('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))" 2>/dev/null || echo cpu)

# --- Install ---

install:
	pip install -e .

install-qwen: install
	pip install -e models/Qwen3-TTS/

install-vallex: install
	pip install -e models/VALL-E-X/

install-vallex-all: install-vallex install-metrics

install-metrics:
	pip install openai-whisper jiwer

install-all: install-qwen install-metrics

install-sox:
	@if command -v apt-get >/dev/null 2>&1; then \
		sudo apt-get install -y sox libsox-dev; \
	elif command -v brew >/dev/null 2>&1; then \
		brew install sox; \
	elif command -v conda >/dev/null 2>&1; then \
		conda install -y -c conda-forge sox; \
	else \
		echo "Please install sox manually: https://sox.sourceforge.net/"; \
	fi

install-cuda:
	pip install torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/cu124

# --- Run ---

run:
	@mkdir -p results
	python models/Qwen3-TTS/benchmarks/benchmark_qwen3tts_real.py --device $(DEVICE) 2>&1 | tee results/benchmark_$(shell date +%Y%m%d_%H%M%S).log

run-vallex:
	@mkdir -p results
	python models/VALL-E-X/benchmarks/benchmark_vallex_real.py --device $(DEVICE) 2>&1 | tee results/benchmark_vallex_$(shell date +%Y%m%d_%H%M%S).log

# Fast pipeline check: short sentences only, 2 per group, no quality metrics.
# Completes in a few minutes. Use to validate the stack before the full run.
smoke-vallex:
	@mkdir -p results
	python models/VALL-E-X/benchmarks/benchmark_vallex_real.py --device $(DEVICE) \
		--groups short --max-per-group 2 --no-quality 2>&1 \
		| tee results/smoke_vallex_$(shell date +%Y%m%d_%H%M%S).log

# torch.profiler run. Default: short sentence, baseline + K4/V2 only (~5-8 min).
# profile_memory/record_shapes are off to keep trace export fast (was previously
# hanging multi-hour on TQ configs under autoregressive decode).
profile-vallex:
	@mkdir -p results
	python models/VALL-E-X/benchmarks/benchmark_vallex_real.py --device $(DEVICE) \
		--profile --profile-sentence short --no-quality 2>&1 \
		| tee results/profile_vallex_$(shell date +%Y%m%d_%H%M%S).log

# Full-coverage profile: medium sentence, all 5 configs. ~30-60 min.
profile-vallex-full:
	@mkdir -p results
	python models/VALL-E-X/benchmarks/benchmark_vallex_real.py --device $(DEVICE) \
		--profile --profile-sentence medium --profile-configs "" --no-quality 2>&1 \
		| tee results/profile_vallex_full_$(shell date +%Y%m%d_%H%M%S).log

evaluate:
	python models/Qwen3-TTS/benchmarks/benchmark_qwen3tts_real.py --evaluate-only

evaluate-vallex:
	python models/VALL-E-X/benchmarks/benchmark_vallex_real.py --evaluate-only

# --- Test ---

test:
	python -m pytest models/Qwen3-TTS/tests/ -v

# --- Cleanup ---

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
