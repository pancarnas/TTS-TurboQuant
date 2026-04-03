.PHONY: install install-qwen install-all install-metrics install-sox install-cuda run evaluate test clean

DEVICE ?= $(shell python -c "import torch; print('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))" 2>/dev/null || echo cpu)

# --- Install ---

install:
	pip install -e .

install-qwen: install
	pip install -e models/Qwen3-TTS/

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

evaluate:
	python models/Qwen3-TTS/benchmarks/benchmark_qwen3tts_real.py --evaluate-only

# --- Test ---

test:
	python -m pytest models/Qwen3-TTS/tests/ -v

# --- Cleanup ---

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
