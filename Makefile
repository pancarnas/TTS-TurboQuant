.PHONY: install install-qwen install-vallex install-vallex-all install-all install-metrics install-sox install-cuda fetch-eval-data validate-eval run run-vallex smoke-vallex profile-vallex profile-vallex-full smoke-qwen profile-qwen profile-qwen-full evaluate evaluate-vallex eval-qwen eval-vallex eval-qwen-parallel eval-vallex-parallel eval-smoke analyze test clean

# Compression-evaluation tunables (see the eval-* targets below).
SEEDS ?= 0,1,2,3,4
DECODE ?= both
TEMPS ?=
GROUPS ?=
# Eval data dir (seed-tts-eval + ellav_hard) — created by `make fetch-eval-data`.
DATA_DIR ?= data
# Parallel workers for eval-*-parallel (NO default — set WORKERS=N explicitly).
WORKERS ?=
# Shared label so a parallel launch's shard CSVs group for `analyze --trials-glob`.
RUN_TAG ?= $(shell date +%Y%m%d_%H%M%S)

DEVICE ?= $(shell python -c "import torch; print('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))" 2>/dev/null || echo cpu)
METRICS_DEVICE ?= $(DEVICE)

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

# --- Eval data ---

# Download seed-tts-eval test-en (HF) + write the ELLA-V hard text into DATA_DIR.
# Required for the seedtts_en / ellav_hard groups (curated smoke/long work offline).
# LIMIT=N fetches only the wavs the first N samples reference (avoids the ~2,170-file
# pull + HF 429); match it to the seedtts_en --max-per-group you plan to run.
fetch-eval-data:
	python tools/fetch_eval_data.py --data-dir $(DATA_DIR) \
		$(if $(LIMIT),--limit $(LIMIT),) $(if $(FETCH_WORKERS),--workers $(FETCH_WORKERS),) \
		$(if $(FETCH_LIBRISPEECH),--fetch-librispeech,) $(if $(FETCH_LIBRITTS),--fetch-libritts,)

# Build the long-context set by concatenating LibriTTS-R (after FETCH_LIBRITTS=1).
build-libritts-long:
	python tools/build_libritts_long.py --data-dir $(DATA_DIR) $(if $(MAX_PER_BUCKET),--max-per-bucket $(MAX_PER_BUCKET),)

# Validate the eval text BEFORE the big run: baseline-only CER per sentence
# (flags floor / un-synthesizable), token-length histogram, reference-clip ASR check.
# Writes results/eval_set_report.md. GROUPS empty = all groups.
validate-eval:
	@mkdir -p results
	python tools/validate_eval_set.py --device $(DEVICE) \
		--metrics-device $(METRICS_DEVICE) --data-dir $(DATA_DIR) $(_GROUPS_ARG) $(_MPG_ARG) $(_MODEL_ARG) \
		--out-md results/eval_set_report.md 2>&1 \
		| tee results/validate_eval_$(shell date +%Y%m%d_%H%M%S).log

# --- Run ---

run:
	@mkdir -p results
	python models/Qwen3-TTS/benchmarks/benchmark_qwen3tts_real.py --device $(DEVICE) 2>&1 | tee results/benchmark_$(shell date +%Y%m%d_%H%M%S).log

# Fast pipeline check for Qwen: short sentences only, 2 per group, no quality metrics.
smoke-qwen:
	@mkdir -p results
	python models/Qwen3-TTS/benchmarks/benchmark_qwen3tts_real.py --device $(DEVICE) \
		--groups smoke --max-per-group 2 --no-quality 2>&1 \
		| tee results/smoke_qwen_$(shell date +%Y%m%d_%H%M%S).log

# torch.profiler run for Qwen. Default: smoke sentence, baseline + K4/V2 (~5-10 min).
profile-qwen:
	@mkdir -p results
	python models/Qwen3-TTS/benchmarks/benchmark_qwen3tts_real.py --device $(DEVICE) \
		--profile --profile-sentence smoke --no-quality 2>&1 \
		| tee results/profile_qwen_$(shell date +%Y%m%d_%H%M%S).log

# Full Qwen profile: long sentence, all 5 configs.
profile-qwen-full:
	@mkdir -p results
	python models/Qwen3-TTS/benchmarks/benchmark_qwen3tts_real.py --device $(DEVICE) \
		--profile --profile-sentence long --profile-configs "" --no-quality 2>&1 \
		| tee results/profile_qwen_full_$(shell date +%Y%m%d_%H%M%S).log

run-vallex:
	@mkdir -p results
	python models/VALL-E-X/benchmarks/benchmark_vallex_real.py --device $(DEVICE) 2>&1 | tee results/benchmark_vallex_$(shell date +%Y%m%d_%H%M%S).log

# Fast pipeline check: short sentences only, 2 per group, no quality metrics.
# Completes in a few minutes. Use to validate the stack before the full run.
smoke-vallex:
	@mkdir -p results
	python models/VALL-E-X/benchmarks/benchmark_vallex_real.py --device $(DEVICE) \
		--groups smoke --max-per-group 2 --no-quality 2>&1 \
		| tee results/smoke_vallex_$(shell date +%Y%m%d_%H%M%S).log

# torch.profiler run. Default: short sentence, baseline + K4/V2 only (~5-8 min).
# profile_memory/record_shapes are off to keep trace export fast (was previously
# hanging multi-hour on TQ configs under autoregressive decode).
profile-vallex:
	@mkdir -p results
	python models/VALL-E-X/benchmarks/benchmark_vallex_real.py --device $(DEVICE) \
		--profile --profile-sentence smoke --no-quality 2>&1 \
		| tee results/profile_vallex_$(shell date +%Y%m%d_%H%M%S).log

# Full-coverage profile: long sentence, all 5 configs. ~30-60 min.
profile-vallex-full:
	@mkdir -p results
	python models/VALL-E-X/benchmarks/benchmark_vallex_real.py --device $(DEVICE) \
		--profile --profile-sentence long --profile-configs "" --no-quality 2>&1 \
		| tee results/profile_vallex_full_$(shell date +%Y%m%d_%H%M%S).log

evaluate:
	python models/Qwen3-TTS/benchmarks/benchmark_qwen3tts_real.py --evaluate-only

evaluate-vallex:
	python models/VALL-E-X/benchmarks/benchmark_vallex_real.py --evaluate-only

# --- Compression evaluation (paired-seed, multi-arm, REAL compression) ---
# Disentangles compression effect from sampling noise. REAL compression is applied
# during decode (track_only=False) so CER/speaker-sim genuinely reflect it. Writes a
# per-trial CSV + KV-reconstruction CSV; analyse with `make analyze`.
# Tunables: SEEDS, DECODE, TEMPS (empty=no sweep), GROUPS (empty=all), METRICS_DEVICE.
#   Temperature sweep:  make eval-qwen TEMPS=0.7,1.0,1.2 DECODE=sampling
#   Scoped/fast:        make eval-qwen GROUPS=hard SEEDS=0,1
_TEMPS_ARG = $(if $(TEMPS),--temperatures $(TEMPS),)
_GROUPS_ARG = $(if $(GROUPS),--groups $(GROUPS),)
_DATA_ARG = $(if $(DATA_DIR),--data-dir $(DATA_DIR),)
# Cap sentences PER GROUP. Important for seedtts_en (1088 rows) — match it to the
# LIMIT you fetched, e.g. MAX_PER_GROUP=100. Smaller groups (ellav_hard=20, long=4)
# are unaffected by a larger cap.
_MPG_ARG = $(if $(MAX_PER_GROUP),--max-per-group $(MAX_PER_GROUP),)
# Residual window: most-recent tokens kept fp16-exact. RW=0 is paper-faithful
# TurboQuant (quantize every token) — makes short/medium sentences compress too.
# Unset → benchmark default (128, the length-gated Exp-1 behavior).
_RW_ARG = $(if $(RW),--residual-window $(RW),)
# WINDOWED=1 emits per-window CER(t)/SpkSim(t) series for long groups (where it
# starts failing) into a windowed sidecar CSV.
_WINDOWED_ARG = $(if $(WINDOWED),--windowed-metrics,)
# Voice strategy (Qwen). MODEL switches checkpoint; VOICE={auto,preset,clone}.
# Preset run (default CustomVoice): leave VOICE unset (auto → preset).
# Clone run: MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-Base VOICE=clone DEFAULT_REF=<wav> DEFAULT_REF_TEXT="..."
_MODEL_ARG = $(if $(MODEL),--model $(MODEL),)
_VOICE_ARG = $(if $(VOICE),--voice-mode $(VOICE),)
_REF_ARG = $(if $(DEFAULT_REF),--default-ref-audio $(DEFAULT_REF),) $(if $(DEFAULT_REF_TEXT),--default-ref-text "$(DEFAULT_REF_TEXT)",)
_QWEN_VOICE_ARGS = $(_MODEL_ARG) $(_VOICE_ARG) $(_REF_ARG)

eval-qwen:
	@mkdir -p results
	python models/Qwen3-TTS/benchmarks/benchmark_qwen3tts_real.py --device $(DEVICE) \
		--metrics-device $(METRICS_DEVICE) --track-only-off \
		--seeds $(SEEDS) --decode $(DECODE) $(_TEMPS_ARG) $(_GROUPS_ARG) $(_DATA_ARG) $(_MPG_ARG) $(_RW_ARG) $(_WINDOWED_ARG) $(_QWEN_VOICE_ARGS) 2>&1 \
		| tee results/eval_qwen_$(shell date +%Y%m%d_%H%M%S).log

eval-vallex:
	@mkdir -p results
	python models/VALL-E-X/benchmarks/benchmark_vallex_real.py --device $(DEVICE) \
		--track-only-off --seeds $(SEEDS) --decode $(DECODE) $(_TEMPS_ARG) $(_GROUPS_ARG) $(_DATA_ARG) $(_MPG_ARG) $(_RW_ARG) 2>&1 \
		| tee results/eval_vallex_$(shell date +%Y%m%d_%H%M%S).log

# Parallel quality sweep: launch WORKERS shards on the single GPU (AR decode is
# latency-bound, so co-location reclaims idle compute). CER/SpkSim are identical to
# a serial run; gather RTF/VRAM separately with WORKERS=1. Shards share RUN_TAG so
# `analyze --trials-glob` reassembles exactly this launch. Set WORKERS=N (e.g. 4).
eval-qwen-parallel:
	@mkdir -p results
	@test -n "$(WORKERS)" || { echo "ERROR: set WORKERS=N (e.g. WORKERS=4)"; exit 1; }
	@echo "Launching $(WORKERS) Qwen shards on $(DEVICE), run-tag=$(RUN_TAG)"
	@for i in $$(seq 0 $$(( $(WORKERS) - 1 ))); do \
		python models/Qwen3-TTS/benchmarks/benchmark_qwen3tts_real.py --device $(DEVICE) \
			--metrics-device $(METRICS_DEVICE) --track-only-off \
			--seeds $(SEEDS) --decode $(DECODE) $(_TEMPS_ARG) $(_GROUPS_ARG) $(_DATA_ARG) $(_MPG_ARG) $(_RW_ARG) $(_WINDOWED_ARG) $(_QWEN_VOICE_ARGS) \
			--num-shards $(WORKERS) --shard-id $$i --run-tag $(RUN_TAG) \
			> results/eval_qwen_shard$${i}_$(RUN_TAG).log 2>&1 & \
	done; \
	wait
	@echo "All shards done. Analyze this run with:"
	@echo "  python tools/analyze_kv_benchmark.py --trials-glob 'results/qwen_trials_shard*_$(RUN_TAG)_*.csv' --out-md results/analysis_$(RUN_TAG).md"

eval-vallex-parallel:
	@mkdir -p results
	@test -n "$(WORKERS)" || { echo "ERROR: set WORKERS=N (e.g. WORKERS=4)"; exit 1; }
	@echo "Launching $(WORKERS) VALL-E shards on $(DEVICE), run-tag=$(RUN_TAG)"
	@for i in $$(seq 0 $$(( $(WORKERS) - 1 ))); do \
		python models/VALL-E-X/benchmarks/benchmark_vallex_real.py --device $(DEVICE) \
			--track-only-off --seeds $(SEEDS) --decode $(DECODE) $(_TEMPS_ARG) $(_GROUPS_ARG) $(_DATA_ARG) $(_MPG_ARG) $(_RW_ARG) \
			--num-shards $(WORKERS) --shard-id $$i --run-tag $(RUN_TAG) \
			> results/eval_vallex_shard$${i}_$(RUN_TAG).log 2>&1 & \
	done; \
	wait
	@echo "All shards done. Analyze this run with:"
	@echo "  python tools/analyze_kv_benchmark.py --trials-glob 'results/vallex_trials_shard*_$(RUN_TAG)_*.csv' --out-md results/analysis_$(RUN_TAG).md"

# Fast pipeline check: 1 long sentence, real compression (no data fetch needed).
# Configs should DIFFER (SpkSim < 1.0) — confirms track_only=False reaches generation.
eval-smoke:
	@mkdir -p results
	python models/Qwen3-TTS/benchmarks/benchmark_qwen3tts_real.py --device $(DEVICE) \
		--metrics-device $(METRICS_DEVICE) --track-only-off $(_RW_ARG) \
		--groups long --max-per-group 1 --seeds 0,1 --decode sampling 2>&1 \
		| tee results/eval_smoke_$(shell date +%Y%m%d_%H%M%S).log

# Statistical analysis of the most recent run: mean±CI, paired Wilcoxon vs baseline,
# CER variance decomposition, temperature + length trends, intrinsic ranking.
# For a parallel run use the --trials-glob command printed by eval-*-parallel.
analyze:
	python tools/analyze_kv_benchmark.py --results-dir results --out-md results/analysis_$(shell date +%Y%m%d_%H%M%S).md

# --- Test ---

test:
	python -m pytest models/Qwen3-TTS/tests/ -v

# --- Cleanup ---

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
