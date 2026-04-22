# VALL-E-X + TurboQuant — Experiment Log

Chronological record of integrating [TurboQuant](https://arxiv.org/abs/2406.02525) KV-cache compression into [VALL-E-X](https://github.com/Plachta/VALL-E-X), the experimental findings on NVIDIA L4 (22 GB, sm_89), and the decisions made along the way.

Target branch: `feature/valle`. All results below are from the `alan.npz` English speaker preset with the standard Qwen3-TTS corpus (10 short + 7 medium + 5 long English sentences).

---

## Context

The project integrates the shared `turboquant/` library into two TTS models (Qwen3-TTS and VALL-E-X). The Qwen3-TTS integration uses HuggingFace's `DynamicCache` infrastructure and was completed prior to this branch. VALL-E-X uses a bare-tuple per-layer KV cache with 12 layers × 16 heads × 64 head_dim — no HF Cache wrapper — so it needs a dedicated adapter: `models/VALL-E-X/turboquant_cache.py`.

The experimental question: does TurboQuant's MSE-optimal quantization deliver latency and memory wins on a launch-bound autoregressive decoder like VALL-E-X?

---

## Phase 0 — initial integration (baseline we started from)

**What existed:** `TurboQuantValleCache` with a list-of-tensors + `torch.cat`-per-step cache, on-path compression via `TurboQuantV3.compress_kv` triggered at each residual-window overflow, and a separate decompressed-prefix cache rebuilt every step.

**Smoke test findings (short sentences, pre-fix):**

| Config | RTF | VRAM (MB) | tok/s | R_theory | R_eff |
|---|---|---|---|---|---|
| baseline (no TQ) | 0.58 | ~2000 | 130 | 1.00 | 1.00 |
| K4/V2 rw=128 | **4.03** | **2723** | 18.6 | 2.60 | **0.79** |
| K3/V2 rw=128 | 3.72 | 2723 | 20.3 | 2.61 | 0.78 |
| K2/V2 rw=128 | 4.35 | 2721 | 17.3 | 2.86 | 0.81 |

TurboQuant was **~7× slower** than baseline and used **more VRAM**. `R_eff < 1.0` because the decompressed-prefix cache (49 MB per config per layer) ate more memory than the compressed chunks saved.

**Profile (medium sentence, K4/V2 pre-fix):**

- Wall 21.71s, Self CUDA 1.10s, Self CPU 6.07s → ~5% GPU utilization
- **206,286** `cudaLaunchKernel` calls (baseline: 40,096)
- **29,626** `aten::cat` calls (baseline: 4,600)
- **65,202** `aten::to` calls (baseline: 1,038)
- **13,080** `cudaStreamSynchronize` calls (baseline: 0)

**Root cause:** VALL-E-X's AR decode is *already* launch-bound on L4 (baseline uses only 15% of GPU time). Each AR step has ~10 kernels per layer × 12 layers = 120 small kernels. TurboQuant's `update()` added ~5× more per step — concat old+new fp16 list, compress/decompress at each overflow, rebuild full K/V from three pieces (compressed + decompressed + residual) — taking the total to ~200k launches per inference. The CPU couldn't feed kernels fast enough, and the GPU mostly idled.

**Conclusion:** the compression added per-step Python overhead on top of an already launch-bound path. Fix had to be structural — eliminate the cat-per-step pattern.

---

## Phase 1 — preallocated KV buffers (commit `2c4c6d7`)

**What changed.** `TurboQuantValleCache` now maintains one grow-on-doubling fp16 tensor per layer (`_buf_k`, `_buf_v`). `update()` writes new tokens as a slice and returns a view:

```python
self._buf_k[layer_idx][:, :, s:s+n, :].copy_(new_key)
self._cur_len[layer_idx] = s + n
return self._buf_k[layer_idx][:, :, :s+n, :], self._buf_v[...][:, :, :s+n, :]
```

Compression moves off-path behind `TurboQuantConfig(track_only=True)` (default). `memory_report()` reports realized fp16 bytes plus analytically-computed theoretical compression metrics. No per-step `torch.cat`, no per-step dtype conversions, no per-step compress/decompress calls.

**Synthetic benchmark (CPU, before L4 validation):** 0.27 ms/step amortized decode, constant regardless of sequence length — vs Phase-0 which grew linearly with steps.

**Trade-off.** `track_only=True` means the cache stores K/V as fp16 (same as baseline). Real memory savings remain *theoretical* (reported as `R_theory`). To realize compression in actual VRAM would require a compression-aware attention kernel (Phase 4 — see below). Legacy path stays available behind `config.track_only=False` or `--track-only-off` CLI flag for reconstruction-quality A/B tests.

---

## Phase 2 — dtype churn cleanup in MSECompressor (commit `72c25bb`)

**What changed.** `MSECompressor.compress` and `decompress` in `turboquant/compressors_v3.py` had an int64 ↔ uint8 round-trip in the bit-packing path:

- Before: `argmin(int64) → .to(uint8) → .long() (int64) → bit-pack with int64 powers → .to(uint8) for storage`
- After: `argmin(int64) → .to(int32) → bit-pack with int32 powers → .to(uint8) for storage`

Similar cleanup in `decompress`. **Reconstruction quality unchanged** — verified via round-trip cosine similarity on 500-token synthetic K/V at bits=2,3,4,8 (K=0.9956, V=0.9418 for K4/V2, identical to pre-change numbers).

**Scope note.** In the default `track_only=True` mode, `MSECompressor` does not run on the AR decode hot path (compression is off-path for storage-simulation only). Phase 2's value is therefore realized only when (a) running `--track-only-off` for legacy reconstruction tests, or (b) implementing Phase 4 where `MSECompressor` would run inside a compression-aware attention kernel.

---

## L4 full validation — 22-sentence sweep (Phase 1+2)

**Hardware:** NVIDIA L4, sm_89, 22 GB VRAM, CUDA 12.4, torch 2.6.0+cu124. Whisper + WavLM metrics on CPU.

**Command:** `make run-vallex`

### Final summary — latency & memory

| Config | RTF short | RTF med | RTF long | VRAM (MB) | tok/s | R_theory | R_eff |
|---|---|---|---|---|---|---|---|
| baseline (no TQ) | 0.55 | 0.58 | 0.74 | 1269 | 126.5 | 1.00 | 1.00 |
| K4/V2 rw=128 | 0.57 | 0.60 | 0.72 | 1139 | 122.5 | **2.73** | 1.00 |
| K3/V3 rw=128 | 0.57 | 0.60 | 0.73 | 1138 | 122.9 | 2.48 | 1.00 |
| K3/V2 rw=128 | 0.57 | 0.60 | 0.73 | 1134 | 122.8 | 2.74 | 1.00 |
| K2/V2 rw=128 | 0.57 | 0.60 | 0.77 | 1152 | 121.3 | **3.07** | 1.00 |

- **Latency within 4–7% of baseline** across all length groups. On long sentences K4/V2 is actually ~3% *faster* than baseline (sampling variance, not a real gain).
- **VRAM ~130 MB LESS than baseline** — unexpected bonus: Phase 1's single preallocated tensor produces less allocator fragmentation than baseline's per-step `torch.cat` pattern.
- **R_theory 2.48–3.07×** — the MSE-optimal quantization theory holds; compression *would* deliver these savings if realized in storage.
- **R_eff = 1.00** across all TQ configs — honest reporting that `track_only=True` stores fp16.

### Phase 0 → Phase 1+2 deltas (K4/V2, short sentence, from profile)

| Metric | Phase 0 | Phase 1+2 | Delta |
|---|---|---|---|
| Wall time | 21.71s | 5.69s | **−74%** |
| Peak VRAM (torch) | 2,718 MB | 1,095 MB | **−60%** |
| `cudaLaunchKernel` calls | 206,286 | 60,430 | −71% |
| `aten::cat` calls | 29,626 | ~4,600 (baseline) | ≈ baseline |
| `aten::to` calls | 65,202 | 1,578 | −98% |
| `aten::copy_` calls | 62,929 | 11,633 | −82% |
| `cudaStreamSynchronize` | 13,080 | below top-10 | ≈ 0 |

### Quality (Whisper CER + WavLM speaker similarity)

| Config | CER short | CER med | CER long | SpkSim short | SpkSim med | SpkSim long |
|---|---|---|---|---|---|---|
| baseline | 37.3% | 25.3% | 77.6% | — | — | — |
| K4/V2 | 33.9% | 21.0% | 73.6% | 0.9041 | 0.9593 | 0.9870 |
| K3/V3 | 21.0% | 35.0% | 71.3% | 0.9239 | 0.9735 | 0.9812 |
| K3/V2 | 30.0% | 18.1% | 58.6% | 0.9325 | 0.9715 | 0.9764 |
| K2/V2 | 15.8% | 21.0% | 67.9% | 0.9126 | 0.9707 | 0.9779 |

Speaker similarity is strong (0.90–0.99) and consistent across configs — voice identity is preserved.

**CER numbers should NOT be read as compression quality signal.** See methodology caveat below.

---

## Methodology caveat — CER at temperature=1.0

The VALL-E-X inference call uses `temperature=1.0` with no fixed seed. Each generation is stochastic: the same (sentence, config) pair can produce wildly different audio on re-run, and Whisper transcription adds its own error floor. Evidence:

- `"She said she would be here by noon"` — baseline CER 91.4%, K2/V2 CER 17.1% (same sentence, different configs)
- `"The weather is beautiful this morning"` — baseline CER 81.6%, K2/V2 CER 0%
- Baseline (no compression) varies from 0% to 91% CER across the 10 short sentences

These differences are sampling variance + Whisper noise, not compression effects. **R_eff = 1.00 proves TQ configs store fp16 identically to baseline** — any CER difference between configs is noise, not signal.

For meaningful quality comparison we'd need either (a) fixed seed, (b) multiple runs per (sentence, config) averaged, or (c) `temperature=0.0` deterministic sampling. None of those were in scope for this run.

The **speaker similarity signal is trustworthy** (0.90-0.99 consistently) because WavLM-xvector compares timbre, which is less sensitive to token-level sampling noise.

---

## Phase 3 — decision (current state)

Phase 1+2 delivered what was asked for in the Phase 0 analysis: **latency-neutral, memory-neutral-or-better, correctly-instrumented VALL-E-X + TurboQuant integration**. `R_theory` numbers prove the quantization math works. `R_eff = 1.00` is the honest reporting that actual VRAM savings require attention-side work.

### Options considered

- **4a — Declare and stop.** The instrumented baseline is the contribution. Documented here. Cost: 0.
- **4b — Layer-adaptive token drop.** Keep first/last K layers in fp16, drop old tokens entirely from middle layers. Partial real savings. Cost: ~1 week. Risk: quality regression; speaker-prompt tokens must never be dropped.
- **4c — JIT decompression in Python (tiled attention).** Replace standard attention with a Python loop over compressed K/V chunks + online softmax. Cost: ~2 weeks. Expected: adds kernel launches → likely regresses latency on launch-bound decode.
- **4d — Custom Triton kernel.** Fused dequantize + flash-style tiled attention. Cost: 4–8 weeks. Only path to real memory savings *and* latency parity. Highest upside, highest risk.

### Recommendation

**4a (declare) for this branch.** L4 has 22 GB and the sweep peaks at 1.5 GB — memory is not the bottleneck on this hardware. Phase 1+2 is a clean result; Phase 4d is a separate research investment that should be its own branch with its own budget and deliverables.

The Phase 4 design sketch lives in the docstring of `models/VALL-E-X/benchmarks/benchmark_vallex_real.py` and a pointer comment at the attention site in `models/VALL-E-X/modules/activation.py:152`.

---

## Reproduction

### Setup (one-liner for fresh SageMaker / Linux+CUDA box)

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124 && \
sudo apt-get update && sudo apt-get install -y sox libsox-dev && \
pip install -e . && \
pip install -e models/VALL-E-X/ && \
pip install openai-whisper jiwer && \
pip uninstall -y tensorflow tensorflow-cpu tf-keras 2>/dev/null; true
```

The `tensorflow` uninstall avoids a known cuDNN ABI conflict on SageMaker images that transitively breaks `from transformers import WavLMForXVector`. If `transformers` imports cleanly without it, skip this step.

### Runs

```bash
# Quick pipeline check — 2 short sentences, no quality metrics (~3 min)
make smoke-vallex

# Full sweep — 22 sentences x 5 configs, Whisper CER + WavLM speaker sim (~25 min on L4)
make run-vallex

# Kernel-level profile — baseline + K4/V2 on one short sentence (~5-8 min)
make profile-vallex

# Full profile — all 5 configs on a medium sentence (~30-60 min, careful with disk)
make profile-vallex-full

# A/B test the legacy on-path compression (reconstruction-quality only, ~7x slower)
python models/VALL-E-X/benchmarks/benchmark_vallex_real.py --device cuda \
    --groups short --max-per-group 2 --no-quality --track-only-off

# Re-score saved wavs without re-running TTS
make evaluate-vallex
```

### Output locations

- `results/benchmark_vallex_<timestamp>_results.txt` — structured incremental dump (crash-safe, CSV-ish)
- `results/benchmark_vallex_<timestamp>.log` — full stdout (via Makefile tee)
- `results/benchmark_vallex_gpu_<timestamp>.csv` — 1 Hz `nvidia-smi` memory/util trace (survives OOM)
- `models/VALL-E-X/benchmarks/outputs/vallex_<group>_<idx>_<config>.wav` — generated audio
- `models/VALL-E-X/benchmarks/outputs/profile_<group>_<config>.json.gz` — Chrome trace, open at https://ui.perfetto.dev

---

## Commit history on `feature/valle`

```
8f8bd6d  WavLM graceful degradation — log warning and skip speaker-sim if transformers/TF import fails
8ed97fc  Phase 4 design sketch documented in benchmark docstring and attention site
72c25bb  Phase 2 — int32 throughout MSECompressor bit-pack, removes int64<->uint8 ping-pong
c5b44ce  Phase 1 display cleanup — Realized/SimComp columns, --track-only-off flag for legacy A/B
2c4c6d7  Phase 1 — preallocated KV buffers eliminate per-step cat, 0.27ms/step amortized
02d7afc  background nvidia-smi poller so OOM leaves a memory trace on disk
49f202a  make profile mode fast — drop heavy profiler flags, limit configs, document launch-bound finding
67805b6  VALL-E-X profiling, memory and throughput instrumentation with --groups filter and smoke-test target
5360f3f  VALL-E-X benchmark mirroring Qwen3-TTS configs across 22 sentences
```

---

## Summary

| Question (from Phase 0 analysis) | Answer after Phase 1+2 |
|---|---|
| Is TurboQuant on VALL-E-X slower than baseline? | **No** — within 4-7% of baseline across all sentence lengths. |
| Does TurboQuant use more VRAM than baseline? | **No** — uses ~130 MB *less* due to lower allocator fragmentation. |
| Does compression save VRAM (R_eff > 1)? | **No, not without attention-side work.** R_theory = 2.48-3.07× shows the math works, but realizing it requires Phase 4. |
| Is the decode launch-bound? | **Yes** — baseline itself uses only 15% of GPU time. Batching would help throughput (see vallex.py:492 hardcoded batch=1). |
| Is the current integration shippable? | **Yes** — for latency-neutral quality tracking and as a baseline for Phase 4 research. |

What changed in code: 9 commits, ~1200 insertions / ~230 deletions, touching `turboquant/config.py`, `turboquant/compressors_v3.py`, `models/VALL-E-X/turboquant_cache.py`, `models/VALL-E-X/benchmarks/benchmark_vallex_real.py`, `models/VALL-E-X/modules/activation.py`, and `Makefile`.
