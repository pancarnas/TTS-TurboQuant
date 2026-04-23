# VALL-E-X vs Qwen3-TTS — TurboQuant Comparison Run Plan

End-to-end recipe for the L4 (or any CUDA box) to compare **VALL-E-X** and **Qwen3-TTS** with and without TurboQuant on medium-length sentences.

Branch: `feature/qwen-phase1` (contains both models' Phase 1 instrumentation + profile mode + nvidia-smi poller + AR/NAR timing split + model-weight VRAM breakout + comparison tool).

---

## 0 — Setup (one-time on a fresh box)

```bash
git fetch origin
git checkout feature/qwen-phase1

# Dependencies (one make invocation, four targets)
make install-cuda install-sox install-all install-vallex

# SageMaker workaround: avoid cuDNN ABI conflict with tensorflow when WavLM loads
pip uninstall -y tensorflow tensorflow-cpu tf-keras 2>/dev/null; true

# Sanity check
python -c "import torch, encodec, vocos, whisper, jiwer; print('cuda:', torch.cuda.is_available())"
```

---

## 1 — Kernel-level profile on medium sentence (Perfetto traces + kernel tables)

Each command produces 2 Chrome traces (baseline + K4/V2) + 1 structured TXT + 1 nvidia-smi CSV.

```bash
# VALL-E-X profile (medium sentence, ~10 min on L4)
python models/VALL-E-X/benchmarks/benchmark_vallex_real.py \
    --device cuda --profile --profile-sentence medium --no-quality

# Qwen3-TTS profile (medium sentence, ~15-20 min on L4 — bigger model)
python models/Qwen3-TTS/benchmarks/benchmark_qwen3tts_real.py \
    --device cuda --profile --profile-sentence medium --no-quality
```

**Output locations:**
| File | Purpose |
|---|---|
| `results/profile_vallex_<ts>.txt` | Structured TXT — GPU config, elapsed, peak VRAM, memory report, top-20 kernels, GPU-bound verdict |
| `results/profile_qwen3tts_<ts>.txt` | Same for Qwen |
| `results/profile_vallex_gpu_<ts>.csv` | 1 Hz `nvidia-smi` — memory.used, utilization, temp (survives OOM) |
| `results/profile_qwen3tts_gpu_<ts>.csv` | Same for Qwen |
| `models/VALL-E-X/benchmarks/outputs/profile_medium_*.json.gz` | 2 × Chrome traces (baseline, K4/V2) |
| `models/Qwen3-TTS/benchmarks/outputs/profile_medium_*.json.gz` | 2 × Chrome traces (baseline, K4/V2) |

---

## 2 — Aggregate metrics on medium group (7 sentences × each config, with quality)

Gives averaged RTF, CER, speaker similarity across 7 medium sentences rather than a single sample.

```bash
# VALL-E-X medium sweep (~8-10 min, 7 sentences × 5 configs)
python models/VALL-E-X/benchmarks/benchmark_vallex_real.py \
    --device cuda --groups medium

# Qwen3-TTS medium sweep (~15-20 min, 7 sentences × 5 configs)
python models/Qwen3-TTS/benchmarks/benchmark_qwen3tts_real.py \
    --device cuda --groups medium
```

**Output:**
| File | Purpose |
|---|---|
| `results/benchmark_vallex_<ts>_results.txt` | Per-sentence CSV + final quality/memory summary |
| `results/benchmark_qwen3tts_<ts>_results.txt` | Same for Qwen |
| `results/benchmark_*_gpu_<ts>.csv` | nvidia-smi timeline |
| `models/*/benchmarks/outputs/*.wav` | All 70 generated wavs (7 × 5 × 2 models) |

Note: the structured TXT now includes per-sentence `ar_rtf` and `nar_rtf` columns for VALL-E (AR/NAR phase split from the new torch.cuda.Event timing).

---

## 3 — Consolidate into a single comparison table

```bash
# Plain text to stdout
python tools/compare_profiles.py results/profile_vallex_*.txt results/profile_qwen3tts_*.txt

# Markdown to a file (good for pasting into reports)
python tools/compare_profiles.py --format markdown results/profile_*.txt > results/comparison.md

# More top kernels visible
python tools/compare_profiles.py --top-n 10 results/profile_*.txt
```

**What it prints:**
- **Headline table**: model × config → wall time, peak VRAM, model weight MB, dynamic MB (peak − weights), realized/sim-compressed MB, R_theory, R_eff, GPU-bound/launch-bound verdict.
- **Top kernels table**: top-N kernels by CUDA time for each (model, config) side-by-side.
- **Environment footer**: GPU name and total VRAM.

---

## 4 — Open Chrome traces in Perfetto

```bash
# Download to local machine first (SageMaker → laptop)
# Option a: SageMaker file browser → right-click → Download
# Option b: scp / s3 cp / etc.
```

Then:
1. Open https://ui.perfetto.dev
2. Drag `.json.gz` onto the page — no decompression needed
3. In the left sidebar, find the `stream 7 7` track (GPU hardware queue) — that's where kernel execution lives
4. Zoom in with `W` key (or mouse wheel) to a ~10 ms window — you'll see individual kernels as colored blocks
5. `Ctrl+F` → search ops like `aten::addmm`, `aten::cat`, `cudaLaunchKernel`
6. For aggregate stats, use the **Query (SQL)** tab:
   ```sql
   SELECT name, COUNT(*) AS cnt, SUM(dur)/1e6 AS total_ms
   FROM slice
   GROUP BY name
   ORDER BY total_ms DESC
   LIMIT 20;
   ```

**What to look for per model:**
| Trace signal | Interpretation |
|---|---|
| Tightly packed kernels on `stream 7 7` | GPU-bound (compute saturated). Expected on Qwen. |
| Visible gaps between tiny kernels | Launch-bound (GPU waiting for CPU). Qwen's profile shouldn't show this after Phase 1. |
| `aten::cat` count (search) | With Phase 1, should be sparse (~12 per layer per step); pre-Phase-1 VALL-E had thousands. |
| `aten::copy_` count | Phase 1 preallocated buffer = 1 copy per layer per step for slice-write; higher is a regression. |

---

## 5 — Headline questions we want answered

After the runs, the comparison table + traces should answer:

1. **Which model is faster in realtime terms on L4?**
   - Prior data: VALL-E RTF ~0.58, Qwen RTF ~1.58 on medium → VALL-E wins 2.7×.
   - Confirm with the wall times in the comparison table.

2. **Which is compute-bound vs launch-bound?**
   - `LAUNCH-BOUND (cuda_time/wall < 50%)` vs `GPU-BOUND`. Prediction: Qwen is GPU-bound (1.7B model, more matmul per step), VALL-E is somewhere between (smaller model, more launch-sensitive).

3. **How much VRAM does each use?**
   - Peak VRAM, split into model weights + dynamic per-generation. Prediction: Qwen ~3.5 GB weights + small dynamic; VALL-E ~0.7 GB weights + ~0.4 GB dynamic.

4. **Is TurboQuant's compression ratio honest?**
   - `R_theory` (2.5–3.1×) vs `R_eff` (1.00 under track_only). Expected — we store fp16 in Phase 1.

5. **What's the per-step overhead from TurboQuant in Phase 1?**
   - Compare `Wall(s)` for baseline vs K4/V2 in the table. Expect <10% difference on both models after Phase 1.

6. **Does the AR vs NAR split on VALL-E show what we expected?**
   - Look at `ar_rtf` + `nar_rtf` columns in `results/benchmark_vallex_<ts>_results.txt`. AR should dominate (~80–95% of wall); NAR is a small fixed cost. Will confirm the "AR is the launch-bound component" finding.

---

## Total time budget

| Step | Wall time on L4 |
|---|---|
| 0. Setup (cold) | 5–10 min |
| 1. VALL-E profile medium | 10 min |
| 1. Qwen profile medium | 15–20 min |
| 2. VALL-E medium sweep | 8–10 min |
| 2. Qwen medium sweep | 15–20 min |
| 3. `compare_profiles.py` | <1 min |
| 4. Trace download + Perfetto viewing | manual, offline |
| **Total compute** | **~55–70 min on L4** |

---

## Troubleshooting

- **WavLM `libcudnn_cnn.so` ABI error**: uninstall tensorflow (see Step 0). Benchmark will degrade gracefully — CER works, speaker-sim skipped with a warning.
- **OOM during Qwen long sentences**: L4 with 22 GB should handle it; if not, use `--groups short,medium` to skip long sentences.
- **Profiler hangs on export**: the Chrome trace file is being written — wait. If it truly hangs >30 min, check the TXT output progression (it's updated incrementally) and Ctrl-C if necessary. Chrome traces for earlier configs are already on disk.
- **Qwen `No module named 'sox'`**: run `make install-qwen` separately — the Qwen test import chain pulls `sox` via `qwen_tts/__init__.py`.
- **`gh auth switch` complaints when pushing**: we're using `pancarnas` account for this repo. If push fails with `403`, run `gh auth switch --user pancarnas` then retry.

---

## Branch state on `feature/qwen-phase1`

Current commits that enable this workflow:
```
82d3fbc  tools/compare_profiles.py
d28664a  model-weight VRAM breakout + VALL-E AR/NAR phase timing split
ee80c00  VALL-E-X docs: realtime-viability section
9764a3a  Qwen profile mode + nvidia-smi poller + crash-safe TXT
2809445  Qwen3-TTS Phase 1 — preallocated KV buffers
f41754a  VALL-E-X experiment log doc (Phase 0/1/2 story)
8f8bd6d  WavLM graceful degradation
72c25bb  Phase 2 — int32 throughout MSECompressor bit-pack
c5b44ce  Phase 1 display cleanup
2c4c6d7  Phase 1 — preallocated KV buffers
```
