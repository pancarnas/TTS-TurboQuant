"""Benchmark TurboQuant KV cache on Qwen3-TTS with real model weights.

Loads the actual Qwen3-TTS model, generates speech with and without
TurboQuant compression, and compares:
  - Latency, peak VRAM, throughput
  - Whisper CER (character error rate)
  - WavLM speaker cosine similarity vs baseline
  - (with --profile) torch.profiler kernel breakdown + Chrome traces

Usage:
    python models/Qwen3-TTS/benchmarks/benchmark_qwen3tts_real.py \
        [--model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice] \
        [--device cuda] [--dtype bfloat16] \
        [--no-quality] [--evaluate-only] \
        [--groups short,medium,long] [--max-per-group N] \
        [--profile] [--profile-sentence {short,medium,long}] [--profile-configs "..."]

Outputs:
    - Per-sentence wavs at models/Qwen3-TTS/benchmarks/outputs/
    - Incremental structured results at
      results/benchmark_qwen3tts_<ts>_results.txt (crash-safe, line-buffered)
    - 1 Hz nvidia-smi memory/util CSV at
      results/benchmark_qwen3tts_gpu_<ts>.csv (survives Python OOM)
    - (with --profile) Chrome traces at
      models/Qwen3-TTS/benchmarks/outputs/profile_<group>_<config>.json.gz
      open at https://ui.perfetto.dev

Requires:
    pip install openai-whisper jiwer
"""

import os
import time
import argparse
import atexit
import datetime
import shutil
import subprocess

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import torch
import soundfile as sf
import librosa

from qwen_tts import Qwen3TTSModel
from qwen_tts.core.models.turboquant_kv_cache import TurboQuantConfig

from turboquant.bench_common import (
    TRIAL_COLUMNS,
    config_bits,
    decode_overrides,
    format_trial_row,
    parse_arms,
    parse_seeds,
    parse_temperatures,
    sentence_hash,
    set_global_seed,
)
from turboquant.eval_sentences import available_groups, iter_eval_items


# --- Test sentences ---
# The evaluation set now lives in turboquant.eval_sentences (shared with the
# VALL-E benchmark for A/B parity): curated `smoke`/`long` groups plus the
# disk-backed `seedtts_en` / `ellav_hard` standard sets. See iter_eval_items.


TURBOQUANT_CONFIGS = [
    ("baseline (no TQ)", None),
    ("K4/V2 rw=128", TurboQuantConfig(key_bits=4, value_bits=2, residual_window=128)),
    ("K3/V3 rw=128", TurboQuantConfig(key_bits=3, value_bits=3, residual_window=128)),
    ("K3/V2 rw=128", TurboQuantConfig(key_bits=3, value_bits=2, residual_window=128)),
    ("K2/V2 rw=128", TurboQuantConfig(key_bits=2, value_bits=2, residual_window=128)),
]


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------


class QualityMetrics:
    """Lazy-loaded quality evaluation: Whisper CER, WavLM speaker sim."""

    def __init__(self, device="cpu"):
        self._device = device
        self._whisper = None
        self._wavlm_model = None
        self._wavlm_extractor = None

    # --- Whisper CER ---

    def _load_whisper(self):
        if self._whisper is None:
            import whisper

            self._whisper = whisper.load_model("base", device=self._device)

    def whisper_cer(
        self, wav: np.ndarray, sr: int, reference_text: str
    ) -> tuple[float, str]:
        """Returns (cer, transcript)."""
        self._load_whisper()
        from jiwer import cer

        # Whisper expects float32 numpy at any sample rate (resamples internally)
        wav = wav.astype(np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)

        result = self._whisper.transcribe(wav)
        transcript = result["text"].strip()

        ref = reference_text.strip()
        hyp = transcript

        if not ref:
            return 0.0, transcript

        error_rate = cer(ref, hyp)
        return float(error_rate), transcript

    # --- WavLM speaker embedding cosine similarity ---

    def _load_wavlm(self):
        if self._wavlm_model is None:
            from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

            self._wavlm_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                "microsoft/wavlm-base-plus-sv"
            )
            self._wavlm_model = (
                WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv")
                .to(self._device)
                .eval()
            )
            # WavLM self-attention is O(audio_len^2); a long clip can OOM the GPU
            # under multi-shard contention. Track its device so we can fall back
            # to CPU permanently on the first OOM rather than dropping a trial row.
            self._wavlm_device = self._device

    def _wavlm_embed(self, inputs: dict) -> np.ndarray:
        inputs = {k: v.to(self._wavlm_device) for k, v in inputs.items()}
        with torch.no_grad():
            emb = self._wavlm_model(**inputs).embeddings
            emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb.squeeze().cpu().numpy()

    def speaker_embedding(self, wav: np.ndarray, sr: int) -> np.ndarray:
        """Extract normalized speaker embedding (512-dim).

        Falls back to CPU permanently on a CUDA OOM (long audio on a contended
        GPU), so a speaker-sim measurement never silently drops a trial row.
        """
        self._load_wavlm()

        wav = wav.astype(np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)

        # WavLM requires 16kHz
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)

        inputs = self._wavlm_extractor(
            wav, sampling_rate=16000, return_tensors="pt", padding=True
        )
        try:
            return self._wavlm_embed(inputs)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            self._wavlm_model = self._wavlm_model.to("cpu")
            self._wavlm_device = "cpu"
            print("[QualityMetrics] WavLM OOM on GPU — moved to CPU for speaker sim.")
            return self._wavlm_embed(inputs)

    def speaker_cosine_similarity(
        self, wav_a: np.ndarray, sr_a: int, wav_b: np.ndarray, sr_b: int
    ) -> float:
        emb_a = self.speaker_embedding(wav_a, sr_a)
        emb_b = self.speaker_embedding(wav_b, sr_b)
        return float(np.dot(emb_a, emb_b))


# ---------------------------------------------------------------------------
# Deterministic intrinsic distortion — compress/decompress real K/V, no sampling
# ---------------------------------------------------------------------------

# Per-layer reconstruction CSV schema (separate grain from the per-trial CSV).
KV_RECON_COLUMNS = [
    "group",
    "idx",
    "sentence_hash",
    "seed",
    "config",
    "variant",
    "layer",
    "cos_k",
    "cos_v",
    "relmse_k",
    "relmse_v",
]


def _extract_fp16_kv(cache):
    """Return (n_layers, fp16_keys, fp16_values) from any cache format, else (0, [], [])."""
    if cache is None:
        return 0, [], []
    # transformers >= 4.57: cache.layers[i].keys/.values
    if (
        hasattr(cache, "layers")
        and len(cache.layers) > 0
        and hasattr(cache.layers[0], "keys")
    ):
        n = len(cache.layers)
        return (
            n,
            [cache.layers[i].keys for i in range(n)],
            [cache.layers[i].values for i in range(n)],
        )
    # older: cache.key_cache[i] / cache.value_cache[i]
    if hasattr(cache, "key_cache") and len(cache.key_cache) > 0:
        n = len(cache.key_cache)
        return (
            n,
            [cache.key_cache[i] for i in range(n)],
            [cache.value_cache[i] for i in range(n)],
        )
    return 0, [], []


def _kv_recon_errors(orig, recon, head_dim):
    """Per-vector mean cosine similarity and relative MSE between fp16 and reconstruction."""
    a = orig.reshape(-1, head_dim).float()
    b = recon.reshape(-1, head_dim).float()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item()
    relmse = (((a - b) ** 2).sum() / (a**2).sum().clamp_min(1e-12)).item()
    return cos, relmse


def _recon_variants():
    """(config_name, key_bits, value_bits, residual_window, variant, prot_layers, prot_bits)
    for every non-baseline config: the production residual setting AND a rw=0
    worst-case bound that compresses every token."""
    out = []
    for cname, tq in TURBOQUANT_CONFIGS:
        if tq is None:
            continue
        kb, vb, rw = config_bits(tq)
        pl = getattr(tq, "protected_layers", 2)
        pb = getattr(tq, "protected_bits", 8)
        out.append((cname, kb, vb, rw, "production", pl, pb))
        out.append((cname, kb, vb, 0, "full", pl, pb))
    return out


def measure_kv_reconstruction(
    model, text, speaker, seed, group_name, idx, shash, kv_fh, results_fh
):
    """Generate one deterministic (greedy) baseline, then compress/decompress the
    SAME fp16 KV cache under every config + variant. Fully deterministic — no
    sampling noise, no CER floor — so it yields the definitive per-config
    compression-distortion ranking. Writes per-layer rows to ``kv_fh`` and
    returns ``{config: {variant: (mean_cos, mean_relmse)}}`` for the summary.
    """
    from turboquant.compressors_v3 import TurboQuantV3

    set_global_seed(seed, deterministic=False)
    model.generate_custom_voice(
        text=text, language="English", speaker=speaker, **decode_overrides("greedy")
    )
    n_layers, fp16_keys, fp16_values = _extract_fp16_kv(
        getattr(model.model, "last_kv_cache", None)
    )
    if n_layers == 0:
        _tee(results_fh, f"    kv_recon [{group_name}:{idx}] ERROR: cannot extract K/V")
        return {}

    head_dim = fp16_keys[0].shape[-1]
    agg = {}
    for cname, kb, vb, rw, variant, pl, pb in _recon_variants():
        cos_acc, relmse_acc = [], []
        for layer_idx in range(n_layers):
            comp = TurboQuantV3(
                head_dim=head_dim,
                key_bits=kb,
                value_bits=vb,
                residual_window=rw,
                layer_idx=layer_idx,
                n_layers=n_layers,
                protected_layers=pl,
                protected_bits=pb,
                seed=42,
                device=str(fp16_keys[layer_idx].device),
            )
            ck, cv = comp.compress_kv(fp16_keys[layer_idx], fp16_values[layer_idx])
            rk, rv = comp.decompress_kv(ck, cv)
            cos_k, relmse_k = _kv_recon_errors(fp16_keys[layer_idx], rk, head_dim)
            cos_v, relmse_v = _kv_recon_errors(fp16_values[layer_idx], rv, head_dim)
            kv_fh.write(
                ",".join(
                    str(x)
                    for x in [
                        group_name,
                        idx,
                        shash,
                        seed,
                        cname,
                        variant,
                        layer_idx,
                        f"{cos_k:.6g}",
                        f"{cos_v:.6g}",
                        f"{relmse_k:.6g}",
                        f"{relmse_v:.6g}",
                    ]
                )
                + "\n"
            )
            cos_acc.append((cos_k + cos_v) / 2)
            relmse_acc.append((relmse_k + relmse_v) / 2)
        agg.setdefault(cname, {})[variant] = (
            sum(cos_acc) / len(cos_acc),
            sum(relmse_acc) / len(relmse_acc),
        )
    return agg


def run_intrinsic_metrics(
    model,
    speaker,
    active_groups,
    max_per_group,
    seed,
    device,
    results_fh,
    data_dir=None,
):
    """Deterministic compression-distortion sweep (KV reconstruction).

    No token sampling is involved, so the resulting ranking is free of the
    variance that makes CER bounce. Writes a per-layer CSV and a console summary.
    """
    kv_fh, kv_path = _open_csv_file("qwen_kv_recon", KV_RECON_COLUMNS)
    _tee(results_fh, f"\n{'=' * 110}")
    _tee(
        results_fh,
        "DETERMINISTIC INTRINSIC DISTORTION — KV reconstruction (no sampling noise)",
    )
    _tee(results_fh, f"Per-layer CSV: {kv_path}")
    _tee(results_fh, f"{'=' * 110}")

    pooled = {}  # config -> variant -> list of (cos, relmse)
    for group_name in active_groups:
        items = iter_eval_items([group_name], max_per_group, data_dir)
        for i, item in enumerate(items):
            text = item.text
            agg = measure_kv_reconstruction(
                model,
                text,
                speaker,
                seed,
                group_name,
                i,
                sentence_hash(text),
                kv_fh,
                results_fh,
            )
            for cname, variants in agg.items():
                for variant, (cos, relmse) in variants.items():
                    pooled.setdefault(cname, {}).setdefault(variant, []).append(
                        (cos, relmse)
                    )
    kv_fh.close()

    _tee(
        results_fh,
        f"\n  {'Config':<14} {'Variant':<12} {'Mean CosSim':<14} {'Mean RelMSE':<14}",
    )
    _tee(results_fh, f"  {'-' * 54}")
    for cname, variants in pooled.items():
        for variant, vals in variants.items():
            mean_cos = sum(c for c, _ in vals) / len(vals)
            mean_relmse = sum(m for _, m in vals) / len(vals)
            _tee(
                results_fh,
                f"  {cname:<14} {variant:<12} {mean_cos:<14.4f} {mean_relmse:<14.4g}",
            )


# ---------------------------------------------------------------------------
# Benchmark logic
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# GPU + memory instrumentation (mirrors VALL-E benchmark)
# ---------------------------------------------------------------------------


def _is_cuda(device) -> bool:
    if isinstance(device, torch.device):
        return device.type == "cuda"
    return isinstance(device, str) and device.startswith("cuda")


def log_gpu_config(device, sink=None):
    """Print device config (name, compute cap, SMs, VRAM, driver) once at run start."""
    lines = ["GPU config:"]
    if not _is_cuda(device):
        lines.append(f"  device={device} (not CUDA — skipping GPU config)")
    else:
        dev = torch.device(device) if isinstance(device, str) else device
        props = torch.cuda.get_device_properties(dev)
        cap = torch.cuda.get_device_capability(dev)
        lines += [
            f"  name={torch.cuda.get_device_name(dev)}",
            f"  compute_capability={cap[0]}.{cap[1]} (sm_{cap[0]}{cap[1]})",
            f"  multi_processor_count={props.multi_processor_count}",
            f"  total_memory={props.total_memory / (1024**3):.1f} GB",
            f"  torch={torch.__version__} cuda={torch.version.cuda}",
        ]
    out = "\n".join(lines)
    print(out)
    if sink is not None:
        sink.write(out + "\n")
        sink.flush()


def read_peak_memory_mb(device) -> float:
    if not _is_cuda(device):
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024**2)


# ---------------------------------------------------------------------------
# Incremental results file (so a crash doesn't lose partial data)
# ---------------------------------------------------------------------------


def _project_root() -> str:
    """Walk up from this file to the repo root (.../TTS-TurboQuant)."""
    # _THIS_DIR = .../models/Qwen3-TTS/benchmarks → 3 dirnames up
    return os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))


def _open_results_file(prefix: str, columns: list | None = None) -> tuple:
    """Open a line-buffered append handle under <repo>/results/. Returns (fh, path)."""
    results_dir = os.path.join(_project_root(), "results")
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(results_dir, f"{prefix}_{ts}.txt")
    fh = open(path, "a", buffering=1, encoding="utf-8")
    fh.write(f"# {prefix} run started {ts}\n")
    if columns:
        fh.write("# columns: " + ",".join(columns) + "\n")
    return fh, path


def _open_csv_file(prefix: str, columns: list) -> tuple:
    """Open a real ``.csv`` (header row, no '#') for downstream pandas analysis."""
    results_dir = os.path.join(_project_root(), "results")
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(results_dir, f"{prefix}_{ts}.csv")
    fh = open(path, "a", buffering=1, encoding="utf-8")
    fh.write(",".join(columns) + "\n")
    return fh, path


def _write_result_line(fh, **kw):
    """Append one per-trial CSV row using the shared TRIAL_COLUMNS schema."""
    fh.write(format_trial_row(kw) + "\n")


def _tee(fh, msg):
    """Print to stdout and to the results file (if any)."""
    print(msg)
    if fh is not None:
        fh.write(msg + "\n")


# ---------------------------------------------------------------------------
# External nvidia-smi poller — survives Python OOM crashes so the last
# memory reading before a crash is preserved on disk.
# ---------------------------------------------------------------------------

_MONITOR_PROC = None
_MONITOR_FH = None


def _stop_nvidia_smi_monitor():
    global _MONITOR_PROC, _MONITOR_FH
    if _MONITOR_PROC is not None:
        try:
            _MONITOR_PROC.terminate()
            try:
                _MONITOR_PROC.wait(timeout=2)
            except subprocess.TimeoutExpired:
                _MONITOR_PROC.kill()
        except Exception:
            pass
        _MONITOR_PROC = None
    if _MONITOR_FH is not None:
        try:
            _MONITOR_FH.flush()
            _MONITOR_FH.close()
        except Exception:
            pass
        _MONITOR_FH = None


def start_nvidia_smi_monitor(prefix: str, interval_s: float = 1.0) -> str:
    """Start a background nvidia-smi polling subprocess.

    Writes CSV (timestamp, memory.used, memory.total, utilization.gpu,
    temperature.gpu) at `interval_s` to results/<prefix>_gpu_<ts>.csv.
    Returns CSV path, or empty string if nvidia-smi is unavailable.
    """
    global _MONITOR_PROC, _MONITOR_FH

    if shutil.which("nvidia-smi") is None or not torch.cuda.is_available():
        return ""

    results_dir = os.path.join(_project_root(), "results")
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(results_dir, f"{prefix}_gpu_{ts}.csv")

    interval_ms = max(int(interval_s * 1000), 100)
    cmd = [
        "nvidia-smi",
        "--query-gpu=timestamp,memory.used,memory.total,utilization.gpu,temperature.gpu",
        "--format=csv,nounits",
        f"-lms={interval_ms}",
    ]
    try:
        fh = open(csv_path, "w", buffering=1, encoding="utf-8")
        fh.write(f"# gpu monitor started {ts}, interval={interval_ms}ms\n")
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"WARNING: could not start nvidia-smi monitor: {e}")
        return ""

    _MONITOR_PROC = proc
    _MONITOR_FH = fh
    atexit.register(_stop_nvidia_smi_monitor)
    return csv_path


# Cached voice-clone capability of the loaded checkpoint: None=unknown, True/False
# once probed. Base models clone; CustomVoice models fall back to the preset speaker.
_CLONE_SUPPORTED = None


def _resolve_voice(item, voice_mode, default_ref_audio, default_ref_text):
    """Effective (ref_audio, ref_text) for an item under a voice mode.

    ``preset`` → never clone. ``clone`` → use the item's reference, else the
    supplied default clip (curated groups have none). ``auto`` → clone when the
    item carries a reference, else preset (run_generation falls back if the
    checkpoint can't clone). Returns (None, None) to mean "use the preset speaker".
    """
    if voice_mode == "preset":
        return None, None
    if voice_mode == "clone":
        if item.ref_audio:
            return item.ref_audio, item.ref_text
        return default_ref_audio, default_ref_text
    return item.ref_audio, item.ref_text  # auto


def run_generation(
    model,
    text,
    language,
    speaker,
    tq_config,
    device=None,
    seed=None,
    gen_overrides=None,
    deterministic=False,
    ref_audio=None,
    ref_text=None,
):
    """Run a single generation.

    Returns (wavs, sr, elapsed, memory_report, peak_vram_mb, n_ar_tokens).
    Wall time covers GPU completion via an explicit sync before/after.

    ``seed`` (if given) is applied via set_global_seed immediately before
    decoding so baseline and every compressed config share the same random
    draw for this (sentence, seed) — the paired-comparison control.
    ``gen_overrides`` merges decode-arm kwargs (e.g. greedy do_sample=False).

    ``ref_audio`` (with optional ``ref_text``) switches to zero-shot voice
    cloning via ``generate_voice_clone``; both clone and preset paths forward
    ``turboquant_config`` to the same ``model.generate``, so compression applies
    identically either way. Without ``ref_audio`` the preset speaker is used.
    """
    kwargs = {}
    if tq_config is not None:
        kwargs["turboquant_config"] = tq_config
    if gen_overrides:
        kwargs.update(gen_overrides)

    if device is None and hasattr(model, "device"):
        device = model.device

    if _is_cuda(device):
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    if seed is not None:
        set_global_seed(seed, deterministic=deterministic)

    if device is None and hasattr(model, "device"):
        device = model.device

    if _is_cuda(device):
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    global _CLONE_SUPPORTED
    start = time.time()
    if ref_audio is not None and _CLONE_SUPPORTED is not False:
        try:
            wavs, sr = model.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=ref_audio,
                ref_text=ref_text,
                **kwargs,
            )
            _CLONE_SUPPORTED = True
        except ValueError as exc:
            if "generate_voice_clone" not in str(exc):
                raise
            # CustomVoice checkpoints reject cloning (custom_voice != base). The
            # check fires before any sampling, so the seed is intact — fall back
            # to the preset speaker. Warn once.
            if _CLONE_SUPPORTED is None:
                print(
                    f"[run_generation] this checkpoint cannot voice-clone — using "
                    f"the preset speaker {speaker!r} instead. (Use a *-Base model "
                    f"for cloning; ground-truth SpkSim will reflect preset mismatch.)"
                )
            _CLONE_SUPPORTED = False
            wavs, sr = model.generate_custom_voice(
                text=text, language=language, speaker=speaker, **kwargs
            )
    else:
        wavs, sr = model.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
            **kwargs,
        )
    if _is_cuda(device):
        torch.cuda.synchronize(device)
    elapsed = time.time() - start

    peak_vram_mb = read_peak_memory_mb(device)

    memory_report = None
    n_ar_tokens = 0
    if hasattr(model.model, "last_kv_cache") and model.model.last_kv_cache is not None:
        cache = model.model.last_kv_cache
        if hasattr(cache, "memory_report"):
            memory_report = cache.memory_report()
        if hasattr(cache, "get_seq_length"):
            try:
                n_ar_tokens = int(cache.get_seq_length(0))
            except Exception:
                n_ar_tokens = 0

    return wavs, sr, elapsed, memory_report, peak_vram_mb, n_ar_tokens


def _empty_group_results() -> dict:
    """Fresh per-config accumulator for one group within one arm."""
    metric_keys = (
        "rtf",
        "cer",
        "spk_sim",
        "spk_sim_ref",
        "theoretical_ratio",
        "effective_ratio",
        "peak_vram_mb",
        "tokens_per_sec",
        "sim_compressed_mb",
        "realized_mb",
    )
    return {name: {k: [] for k in metric_keys} for name, _ in TURBOQUANT_CONFIGS}


def _load_reference_clip(item, metrics):
    """Load the reference clip for ground-truth SpkSim: ground truth, else prompt.

    Returns ``(wav_np, sr)`` or ``(None, None)`` when there is no reference or no
    metrics. Loaded once per cell so it isn't re-read for every config.
    """
    ref_path = getattr(item, "ground_truth_audio", None) or getattr(
        item, "ref_audio", None
    )
    if not ref_path or metrics is None:
        return None, None
    try:
        wav, sr = librosa.load(ref_path, sr=None, mono=True)
        return wav, sr
    except Exception:
        return None, None


def _extract_mem_metrics(mem_report) -> dict:
    """Pull compression ratios + byte counts (in MB) out of a cache memory_report."""
    if not mem_report:
        return {
            "theoretical_ratio": 1.0,
            "effective_ratio": 1.0,
            "sim_compressed_mb": 0.0,
            "realized_mb": 0.0,
        }
    theoretical = mem_report.get(
        "theoretical_compression_ratio", mem_report.get("compression_ratio", 1.0)
    )
    effective = mem_report.get("effective_compression_ratio", theoretical)
    return {
        "theoretical_ratio": theoretical,
        "effective_ratio": effective,
        "sim_compressed_mb": mem_report.get("compressed_bytes", 0) / (1024**2),
        "realized_mb": mem_report.get(
            "realized_fp16_bytes", mem_report.get("total_bytes", 0)
        )
        / (1024**2),
    }


def _sweep_arm(
    model,
    metrics,
    speaker,
    arm,
    active_groups,
    max_per_group,
    seeds,
    temperatures,
    output_dir,
    results_fh,
    trial_fh,
    device,
    deterministic,
    data_dir=None,
    num_shards=1,
    shard_id=0,
    voice_mode="auto",
    default_ref_audio=None,
    default_ref_text=None,
):
    """Run the full group×sentence×seed×temperature×config sweep for one arm.

    For each (sentence, seed, temperature) all configs share that seed's random
    draw (paired design). Writes one trial row per
    (arm, sentence, seed, temperature, config) to ``trial_fh`` and returns
    ``summary[group][config]`` of metric lists for the console tables.

    Temperature only varies for the sampling arm (greedy ignores it, so it uses a
    single ``None`` anchor).
    """
    temps = temperatures if (arm == "sampling" and temperatures) else [None]
    _tee(results_fh, f"\n{'#' * 110}")
    base_overrides = decode_overrides(arm)
    temp_note = f" temps={temps}" if temps != [None] else ""
    _tee(
        results_fh,
        f"DECODE ARM: {arm}  (overrides={base_overrides or 'model defaults'}){temp_note}",
    )
    _tee(results_fh, f"{'#' * 110}")

    summary = {}
    # Round-robin shard counter over (group, sentence, seed, temperature) cells.
    # The config loop runs whole inside each cell, so a baseline always stays in
    # the same process as the configs it anchors (baseline-as-ref SpkSim).
    cell_idx = 0
    for group_name in active_groups:
        items = iter_eval_items([group_name], max_per_group, data_dir)
        n = len(items)
        _tee(results_fh, f"\n{'=' * 110}")
        _tee(
            results_fh,
            f"[{arm}] Group: {group_name} "
            f"({n} sentences × {len(seeds)} seeds × {len(temps)} temps)",
        )
        _tee(results_fh, f"{'=' * 110}")

        group_results = _empty_group_results()
        for i, item in enumerate(items):
            text = item.text
            shash = sentence_hash(text)
            preview = text[:50] + "..." if len(text) > 50 else text
            _tee(results_fh, f'\n  [{i + 1}/{n}] "{preview}"')
            for seed in seeds:
                for temp in temps:
                    this_cell = cell_idx
                    cell_idx += 1
                    if this_cell % num_shards != shard_id:
                        continue
                    gen_overrides = decode_overrides(arm, temperature=temp)
                    _sweep_sentence_seed(
                        model,
                        metrics,
                        speaker,
                        item,
                        group_name,
                        i,
                        shash,
                        arm,
                        seed,
                        temp,
                        gen_overrides,
                        group_results,
                        output_dir,
                        results_fh,
                        trial_fh,
                        device,
                        deterministic,
                        save_wav=(seed == seeds[0] and temp == temps[0]),
                        voice_mode=voice_mode,
                        default_ref_audio=default_ref_audio,
                        default_ref_text=default_ref_text,
                    )

        _print_group_averages(results_fh, metrics, group_name, group_results)
        summary[group_name] = group_results
    return summary


def _sweep_sentence_seed(
    model,
    metrics,
    speaker,
    item,
    group_name,
    idx,
    shash,
    arm,
    seed,
    temperature,
    gen_overrides,
    group_results,
    output_dir,
    results_fh,
    trial_fh,
    device,
    deterministic,
    save_wav,
    voice_mode="auto",
    default_ref_audio=None,
    default_ref_text=None,
):
    """Run all configs for one (sentence, seed, temperature), paired on the same draw."""
    text = item.text
    baseline_wav = None
    baseline_sr = None
    eff_ref_audio, eff_ref_text = _resolve_voice(
        item, voice_mode, default_ref_audio, default_ref_text
    )
    # In clone mode an item with no reference (and no default clip) can't be voiced.
    if voice_mode == "clone" and not eff_ref_audio:
        _tee(results_fh, f"    skip (clone mode, no reference) {group_name}/{idx}")
        return
    # Ground-truth reference clip (same text) preferred; else the cloning prompt.
    # Loaded once per cell for the side-by-side spk_sim_ref metric.
    ref_wav, ref_sr = _load_reference_clip(item, metrics)
    for config_name, tq_config in TURBOQUANT_CONFIGS:
        key_bits, value_bits, residual_window = config_bits(tq_config)
        try:
            wavs, sr, elapsed, mem_report, peak_vram_mb, n_ar_tokens = run_generation(
                model,
                text,
                "English",
                speaker,
                tq_config,
                device=device,
                seed=seed,
                gen_overrides=gen_overrides,
                deterministic=deterministic,
                ref_audio=eff_ref_audio,
                ref_text=eff_ref_text,
            )
            wav = wavs[0]
            audio_duration = len(wav) / sr
            rtf = elapsed / audio_duration if audio_duration > 0 else float("inf")
            tokens_per_sec = n_ar_tokens / elapsed if elapsed > 0 else 0.0
            mm = _extract_mem_metrics(mem_report)

            r = group_results[config_name]
            r["rtf"].append(rtf)
            r["peak_vram_mb"].append(peak_vram_mb)
            r["tokens_per_sec"].append(tokens_per_sec)
            if mem_report:
                for k in (
                    "theoretical_ratio",
                    "effective_ratio",
                    "sim_compressed_mb",
                    "realized_mb",
                ):
                    r[k].append(mm[k])

            error_rate = None
            spk_sim = None
            spk_sim_ref = None
            transcript_len = None
            if metrics:
                error_rate, transcript = metrics.whisper_cer(wav, sr, text)
                transcript_len = len(transcript)
                r["cer"].append(error_rate)
                if tq_config is None:
                    baseline_wav, baseline_sr = wav, sr
                elif baseline_wav is not None:
                    spk_sim = metrics.speaker_cosine_similarity(
                        baseline_wav, baseline_sr, wav, sr
                    )
                    if spk_sim is not None:
                        r["spk_sim"].append(spk_sim)
                # Ground-truth SIM (vs an external reference clip) — reported for
                # every config incl. baseline, since the reference is not the run.
                if ref_wav is not None:
                    spk_sim_ref = metrics.speaker_cosine_similarity(
                        ref_wav, ref_sr, wav, sr
                    )
                    if spk_sim_ref is not None:
                        r["spk_sim_ref"].append(spk_sim_ref)

            if save_wav:
                safe = config_name.replace(" ", "_").replace("/", "_")
                tsuf = "" if temperature is None else f"_t{temperature}"
                out_path = os.path.join(
                    output_dir,
                    f"qwen_{group_name}_{idx}_{arm}_s{seed}{tsuf}_{safe}.wav",
                )
                sf.write(out_path, wav, sr)

            status = (
                f"RTF={rtf:.2f} VRAM={peak_vram_mb:.0f}MB tok/s={tokens_per_sec:.1f}"
            )
            if error_rate is not None:
                status += f" CER={error_rate:.1%}"
                if spk_sim is not None:
                    status += f" SpkSim={spk_sim:.4f}"
                if spk_sim_ref is not None:
                    status += f" SpkSimRef={spk_sim_ref:.4f}"
            tlabel = "" if temperature is None else f" T={temperature}"
            _tee(results_fh, f"    s{seed}{tlabel} {config_name:<22} {status}")

            _write_result_line(
                trial_fh,
                arm=arm,
                seed=seed,
                temperature=temperature,
                group=group_name,
                idx=idx,
                sentence_hash=shash,
                config=config_name,
                key_bits=key_bits,
                value_bits=value_bits,
                residual_window=residual_window,
                rtf=rtf,
                cer=error_rate,
                transcript_len=transcript_len,
                spk_sim=spk_sim,
                spk_sim_ref=spk_sim_ref,
                peak_vram_mb=peak_vram_mb,
                tokens_per_sec=tokens_per_sec,
                n_ar_tokens=n_ar_tokens,
                sim_compressed_mb=mm["sim_compressed_mb"],
                realized_mb=mm["realized_mb"],
                theoretical_ratio=mm["theoretical_ratio"],
                effective_ratio=mm["effective_ratio"],
            )
        except Exception as e:
            _tee(results_fh, f"    s{seed} {config_name:<22} ERROR: {e}")
            import traceback

            traceback.print_exc()


def _print_group_averages(results_fh, metrics, group_name, group_results) -> None:
    """Console-friendly per-config means for one group (pooled over sentences×seeds)."""

    def _avg(xs, default=0.0):
        return sum(xs) / len(xs) if xs else default

    _tee(results_fh, f"\n  {'─' * 80}")
    _tee(results_fh, f"  AVERAGES for {group_name} (pooled over sentences × seeds):")
    _tee(results_fh, f"  {'─' * 80}")
    if metrics:
        _tee(
            results_fh,
            f"  {'Config':<22} {'RTF':<7} {'CER':<8} {'SpkSim':<8} "
            f"{'VRAM(MB)':<10} {'tok/s':<8} {'R_th':<6} {'R_eff':<6}",
        )
    else:
        _tee(
            results_fh,
            f"  {'Config':<22} {'RTF':<7} {'VRAM(MB)':<10} {'tok/s':<8} "
            f"{'R_th':<6} {'R_eff':<6}",
        )
    _tee(results_fh, f"  {'-' * 90}")
    for config_name, tq_config in TURBOQUANT_CONFIGS:
        r = group_results[config_name]
        avg_rtf = _avg(r["rtf"])
        avg_vram = _avg(r["peak_vram_mb"])
        avg_tps = _avg(r["tokens_per_sec"])
        avg_th = _avg(r["theoretical_ratio"], default=1.0 if tq_config is None else 0.0)
        avg_eff = _avg(r["effective_ratio"], default=1.0 if tq_config is None else 0.0)
        if metrics:
            avg_cer = _avg(r["cer"])
            avg_spk = _avg(r["spk_sim"])
            spk_str = f"{avg_spk:.4f}" if tq_config is not None else "---"
            _tee(
                results_fh,
                f"  {config_name:<22} {avg_rtf:<7.2f} {avg_cer:<8.2%} "
                f"{spk_str:<8} {avg_vram:<10.0f} {avg_tps:<8.1f} "
                f"{avg_th:<6.2f} {avg_eff:<6.2f}",
            )
        else:
            _tee(
                results_fh,
                f"  {config_name:<22} {avg_rtf:<7.2f} {avg_vram:<10.0f} "
                f"{avg_tps:<8.1f} {avg_th:<6.2f} {avg_eff:<6.2f}",
            )


def benchmark_qwen3tts(args):
    device = args.device
    arms = parse_arms(args.decode)
    seeds = args.seeds

    num_shards = getattr(args, "num_shards", 1)
    shard_id = getattr(args, "shard_id", 0)
    data_dir = getattr(args, "data_dir", None)
    run_tag = getattr(args, "run_tag", "") or ""
    # Each parallel worker writes its own shard CSV so workers never collide; the
    # shared run_tag lets analyze concat one launch's shards (and only those).
    trial_tag = "qwen_trials" if num_shards == 1 else f"qwen_trials_shard{shard_id}"
    if run_tag:
        trial_tag = f"{trial_tag}_{run_tag}"

    results_fh, results_path = _open_results_file("benchmark_qwen3tts")
    trial_fh, trial_path = _open_csv_file(trial_tag, TRIAL_COLUMNS)
    gpu_csv = start_nvidia_smi_monitor("benchmark_qwen3tts")

    header = [
        "=" * 110,
        "Qwen3-TTS Rigorous Benchmark (paired-seed, multi-arm)",
        f"Model: {args.model}",
        f"Device: {device} | Dtype: {args.dtype} | Quality metrics: {not args.no_quality}",
        f"Decode arms: {arms} | Seeds: {seeds} | Deterministic: {args.deterministic}",
        (
            f"Temperature sweep (sampling arm): {args.temperatures or 'off (model default)'}"
        ),
        (
            f"Compression mode: {'REAL (track_only=False)' if args.track_only_off else 'TRACK-ONLY (fp16; configs produce IDENTICAL audio — CER/SpkSim cannot differ)'}"
        ),
        "Pairing: each (sentence, seed) reseeds RNG identically before every config,",
        "  so baseline and compressed configs share one random draw — divergence is",
        "  the compression signal, not seed luck. Per-trial CSV is the source of truth.",
        f"Human log: {results_path}",
        f"Per-trial CSV: {trial_path}",
        f"GPU monitor CSV: {gpu_csv or '(nvidia-smi unavailable — skipped)'}",
        "=" * 110,
    ]
    for line in header:
        _tee(results_fh, line)

    log_gpu_config(device, sink=results_fh)

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(args.dtype, torch.bfloat16)

    _tee(results_fh, "\nLoading Qwen3-TTS model...")
    t0 = time.time()
    if _is_cuda(device):
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    model = Qwen3TTSModel.from_pretrained(
        args.model, device_map=args.device, dtype=dtype
    )
    if _is_cuda(device):
        torch.cuda.synchronize(device)
    _tee(results_fh, f"Model loaded in {time.time() - t0:.1f}s")
    model_weight_mb = read_peak_memory_mb(device)
    _tee(results_fh, f"Model weights + load-time VRAM: {model_weight_mb:.0f} MB")

    speakers = model.get_supported_speakers()
    speaker = speakers[0] if speakers else "Ryan"
    _tee(results_fh, f"Using speaker: {speaker}")

    metrics = None
    if not args.no_quality:
        _tee(results_fh, f"\nLoading quality metrics on {args.metrics_device}...")
        metrics = QualityMetrics(device=args.metrics_device)

    # One warmup per config so TurboQuant lazy-init / CUDA kernel autotune
    # doesn't skew the first-sentence measurement.
    _tee(results_fh, "\nWarmup (one per config)...")
    for config_name, tq_config in TURBOQUANT_CONFIGS:
        try:
            run_generation(
                model, "Hello world.", "English", speaker, tq_config, device=device
            )
        except Exception as e:
            _tee(results_fh, f"  warmup {config_name}: ERROR {e}")
    _tee(results_fh, "Warmup done.\n")

    output_dir = os.path.join(_THIS_DIR, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    active_groups = getattr(args, "active_groups", available_groups())
    max_per_group = getattr(args, "max_per_group", None)

    for arm in arms:
        summary = _sweep_arm(
            model,
            metrics,
            speaker,
            arm,
            active_groups,
            max_per_group,
            seeds,
            args.temperatures,
            output_dir,
            results_fh,
            trial_fh,
            device,
            args.deterministic,
            data_dir=data_dir,
            num_shards=num_shards,
            shard_id=shard_id,
            voice_mode=getattr(args, "voice_mode", "auto"),
            default_ref_audio=getattr(args, "default_ref_audio", None),
            default_ref_text=getattr(args, "default_ref_text", None),
        )
        _print_arm_summaries(results_fh, metrics, arm, active_groups, summary)

    trial_fh.close()
    _tee(results_fh, f"\nOutput audio saved to: {output_dir}/")
    _tee(results_fh, f"Per-trial CSV (source of truth): {trial_path}")

    # Deterministic intrinsic distortion metrics (no sampling noise, no CER floor).
    # Only the first shard runs it (it is deterministic and not sharded).
    if not args.no_intrinsic and shard_id == 0:
        run_intrinsic_metrics(
            model,
            speaker,
            active_groups,
            max_per_group,
            seeds[0],
            device,
            results_fh,
            data_dir=data_dir,
        )

    results_fh.close()


def _print_arm_summaries(results_fh, metrics, arm, active_groups, summary) -> None:
    """Final per-arm quality + memory/throughput tables (means across all sentences)."""
    if metrics:
        _tee(results_fh, f"\n{'=' * 110}")
        _tee(
            results_fh,
            f"[{arm}] FINAL SUMMARY — Quality (means across sentences × seeds)",
        )
        _tee(results_fh, f"{'=' * 110}")
        header_row = f"{'Config':<22} "
        for group_name in active_groups:
            header_row += f"{'RTF':<7} {'CER':<7} {'SpkSim':<9} "
        _tee(results_fh, header_row)
        for config_name, tq_config in TURBOQUANT_CONFIGS:
            row = f"{config_name:<22} "
            for group_name in active_groups:
                r = summary[group_name][config_name]
                avg_rtf = sum(r["rtf"]) / len(r["rtf"]) if r["rtf"] else 0
                avg_cer = sum(r["cer"]) / len(r["cer"]) if r["cer"] else 0
                avg_spk = sum(r["spk_sim"]) / len(r["spk_sim"]) if r["spk_sim"] else 0
                spk_str = f"{avg_spk:.4f}" if tq_config is not None else "---"
                row += f"{avg_rtf:<7.2f} {avg_cer:<7.1%} {spk_str:<9} "
            _tee(results_fh, row)

    _tee(results_fh, f"\n{'=' * 110}")
    _tee(results_fh, f"[{arm}] FINAL SUMMARY — Memory & Throughput")
    _tee(results_fh, f"{'=' * 110}")
    _tee(
        results_fh,
        f"{'Config':<22} {'VRAM(MB)':<10} {'tok/s':<8} {'Realized(MB)':<14} "
        f"{'SimComp(MB)':<14} {'R_theory':<10} {'R_eff':<8}",
    )
    _tee(results_fh, "-" * 100)
    for config_name, tq_config in TURBOQUANT_CONFIGS:
        vram, tps, realized, sim_comp, th, eff = [], [], [], [], [], []
        for group_name in active_groups:
            r = summary[group_name][config_name]
            vram += r["peak_vram_mb"]
            tps += r["tokens_per_sec"]
            realized += r["realized_mb"]
            sim_comp += r["sim_compressed_mb"]
            th += r["theoretical_ratio"]
            eff += r["effective_ratio"]
        avg_vram = sum(vram) / len(vram) if vram else 0
        avg_tps = sum(tps) / len(tps) if tps else 0
        avg_realized = sum(realized) / len(realized) if realized else 0
        avg_sim = sum(sim_comp) / len(sim_comp) if sim_comp else 0
        avg_th = sum(th) / len(th) if th else (1.0 if tq_config is None else 0)
        avg_eff = sum(eff) / len(eff) if eff else (1.0 if tq_config is None else 0)
        _tee(
            results_fh,
            f"{config_name:<22} {avg_vram:<10.0f} {avg_tps:<8.1f} "
            f"{avg_realized:<14.2f} {avg_sim:<14.2f} {avg_th:<10.2f} {avg_eff:<8.2f}",
        )


# ---------------------------------------------------------------------------
# Profiling mode (mirrors VALL-E benchmark)
# ---------------------------------------------------------------------------


@torch.no_grad()
def profile_generation(
    model,
    text,
    language,
    speaker,
    tq_config,
    device,
    output_dir,
    config_name,
    group_name,
):
    """Profile a single Qwen generation with torch.profiler.
    Returns (prof, elapsed, trace_path)."""
    from torch.profiler import profile, ProfilerActivity

    activities = [ProfilerActivity.CPU]
    if _is_cuda(device):
        activities.append(ProfilerActivity.CUDA)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    kwargs = {}
    if tq_config is not None:
        kwargs["turboquant_config"] = tq_config

    start = time.time()
    # Lightweight profiler flags — record_shapes / profile_memory generate huge
    # traces on autoregressive decode and make export_chrome_trace take hours.
    with profile(
        activities=activities,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        model.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
            **kwargs,
        )
    if _is_cuda(device):
        torch.cuda.synchronize(device)
    elapsed = time.time() - start

    safe = config_name.replace(" ", "_").replace("/", "_")
    trace_path = os.path.join(output_dir, f"profile_{group_name}_{safe}.json.gz")
    prof.export_chrome_trace(trace_path)
    return prof, elapsed, trace_path


def _profile_ratio_classification(prof, elapsed_s: float) -> tuple[float, str]:
    """GPU-bound vs launch-bound classifier. Returns (cuda_time_fraction, label)."""
    try:
        total_cuda_us = sum(ev.cuda_time_total for ev in prof.key_averages())
    except Exception:
        total_cuda_us = 0
    wall_us = elapsed_s * 1e6
    frac = (total_cuda_us / wall_us) if wall_us > 0 else 0.0
    if frac >= 0.5:
        label = f"GPU-BOUND (cuda_time/wall = {frac:.1%}): batching won't materially help throughput."
    else:
        label = (
            f"LAUNCH-BOUND (cuda_time/wall = {frac:.1%}): batching would likely help. "
            f"Inspect Chrome trace for the dominant launch cost."
        )
    return frac, label


def profile_all_configs(args):
    """Profile one representative sentence per TurboQuant config."""
    device = args.device
    results_fh, results_path = _open_results_file("profile_qwen3tts")
    gpu_csv = start_nvidia_smi_monitor("profile_qwen3tts")

    header = [
        "=" * 110,
        "Qwen3-TTS Profile Run",
        f"Model: {args.model}",
        f"Device: {device} | Dtype: {args.dtype} | profile_sentence={args.profile_sentence}",
        f"Structured results: {results_path}",
        f"GPU monitor CSV: {gpu_csv or '(nvidia-smi unavailable — skipped)'}",
        "=" * 110,
    ]
    for line in header:
        _tee(results_fh, line)
    log_gpu_config(device, sink=results_fh)

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(args.dtype, torch.bfloat16)

    _tee(results_fh, "\nLoading Qwen3-TTS model...")
    t0 = time.time()
    if _is_cuda(device):
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    model = Qwen3TTSModel.from_pretrained(args.model, device_map=device, dtype=dtype)
    if _is_cuda(device):
        torch.cuda.synchronize(device)
    _tee(results_fh, f"Model loaded in {time.time() - t0:.1f}s")
    model_weight_mb = read_peak_memory_mb(device)
    _tee(results_fh, f"Model weights + load-time VRAM: {model_weight_mb:.0f} MB")

    speakers = model.get_supported_speakers()
    speaker = speakers[0] if speakers else "Ryan"
    _tee(results_fh, f"Using speaker: {speaker}")

    text = iter_eval_items(
        [args.profile_sentence],
        max_per_group=1,
        data_dir=getattr(args, "data_dir", None),
    )[0].text
    _tee(results_fh, f'Profiling text ({args.profile_sentence}): "{text[:80]}..."')

    output_dir = os.path.join(_THIS_DIR, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Filter which configs to profile (default: baseline + K4/V2).
    requested = getattr(args, "profile_configs", None)
    if requested:
        names = [c.strip() for c in requested.split(",") if c.strip()]
        configs_to_run = [(n, c) for (n, c) in TURBOQUANT_CONFIGS if n in names]
        if not configs_to_run:
            _tee(
                results_fh,
                f"ERROR: --profile-configs matched nothing. "
                f"Requested: {names}. Available: {[n for n, _ in TURBOQUANT_CONFIGS]}",
            )
            results_fh.close()
            return
    else:
        configs_to_run = TURBOQUANT_CONFIGS

    # Per-config warmup
    _tee(results_fh, "\nWarmup (one per config)...")
    for config_name, tq_config in configs_to_run:
        try:
            run_generation(
                model, "Hello world.", "English", speaker, tq_config, device=device
            )
        except Exception as e:
            _tee(results_fh, f"  warmup {config_name}: ERROR {e}")
    _tee(results_fh, "Warmup done.\n")

    for config_name, tq_config in configs_to_run:
        _tee(results_fh, f"\n{'=' * 110}")
        _tee(results_fh, f"Profiling config: {config_name}")
        _tee(results_fh, f"{'=' * 110}")
        try:
            prof, elapsed, trace_path = profile_generation(
                model,
                text,
                "English",
                speaker,
                tq_config,
                device,
                output_dir,
                config_name,
                args.profile_sentence,
            )

            peak_vram_mb = read_peak_memory_mb(device)
            mem_report = None
            if (
                hasattr(model.model, "last_kv_cache")
                and model.model.last_kv_cache is not None
            ):
                cache = model.model.last_kv_cache
                if hasattr(cache, "memory_report"):
                    mem_report = cache.memory_report()

            _tee(
                results_fh, f"Elapsed: {elapsed:.2f}s  Peak VRAM: {peak_vram_mb:.0f} MB"
            )
            if mem_report:
                comp_mb = mem_report.get("compressed_bytes", 0) / 1024**2
                real_mb = (
                    mem_report.get(
                        "realized_fp16_bytes", mem_report.get("total_bytes", 0)
                    )
                    / 1024**2
                )
                _tee(
                    results_fh,
                    f"Realized: {real_mb:.2f} MB | SimComp: {comp_mb:.2f} MB | "
                    f"R_theory={mem_report.get('theoretical_compression_ratio', mem_report.get('compression_ratio', 1.0)):.2f}x "
                    f"R_eff={mem_report.get('effective_compression_ratio', 1.0):.2f}x",
                )

            frac, label = _profile_ratio_classification(prof, elapsed)
            _tee(results_fh, label)

            sort_key = "cuda_time_total" if _is_cuda(device) else "self_cpu_time_total"
            try:
                table = prof.key_averages().table(sort_by=sort_key, row_limit=20)
            except Exception as e:
                table = f"(key_averages failed: {e})"
            _tee(results_fh, f"\nTop-20 by {sort_key}:")
            _tee(results_fh, table)

            try:
                cpu_table = prof.key_averages().table(
                    sort_by="self_cpu_time_total", row_limit=10
                )
            except Exception as e:
                cpu_table = f"(cpu table failed: {e})"
            _tee(results_fh, "\nTop-10 by self_cpu_time_total:")
            _tee(results_fh, cpu_table)

            _tee(results_fh, f"\nChrome trace: {trace_path}")
        except Exception as e:
            _tee(results_fh, f"Profile {config_name}: ERROR {e}")
            import traceback

            traceback.print_exc()
        finally:
            # Release the profiler (retains ~1KB of metadata per kernel event —
            # for Qwen that's ~700MB per config of host RAM) and the model's
            # cached KV cache before moving to the next config. Without this,
            # the second config's profile can OOM the host.
            import gc

            try:
                del prof
            except NameError:
                pass
            if hasattr(model.model, "last_kv_cache"):
                model.model.last_kv_cache = None
            gc.collect()
            if _is_cuda(device):
                torch.cuda.empty_cache()

    _tee(results_fh, f"\nStructured results: {results_path}")
    results_fh.close()


def evaluate_saved_wavs(args):
    """Evaluate quality metrics on previously saved wav files (no TTS model needed)."""
    output_dir = os.path.join(_THIS_DIR, "outputs")
    if not os.path.exists(output_dir):
        print(
            f"ERROR: No outputs found at {output_dir}. Run generation first (without --evaluate-only)."
        )
        return

    print("=" * 80)
    print("Qwen3-TTS Quality Evaluation (from saved wavs)")
    print("=" * 80)

    metrics = QualityMetrics(device="cpu")

    active_groups = getattr(args, "active_groups", available_groups())
    data_dir = getattr(args, "data_dir", None)
    max_per_group = getattr(args, "max_per_group", None)
    for group_name in active_groups:
        items = iter_eval_items([group_name], max_per_group, data_dir)
        print(f"\n{'─' * 80}")
        print(f"Group: {group_name} ({len(items)} sentences)")
        print(f"{'─' * 80}")
        print(f"{'Config':<22} {'Avg CER':<10} {'Avg SpkSim':<12}")
        print("-" * 45)

        group_results = {
            name: {"cer": [], "spk_sim": []} for name, _ in TURBOQUANT_CONFIGS
        }

        for i, item in enumerate(items):
            text = item.text
            baseline_wav = None
            baseline_sr = None
            for config_name, tq_config in TURBOQUANT_CONFIGS:
                out_path = os.path.join(
                    output_dir,
                    f"qwen_{group_name}_{i}_{config_name.replace(' ', '_').replace('/', '_')}.wav",
                )
                if not os.path.exists(out_path):
                    continue
                wav, sr = sf.read(out_path)
                wav = wav.astype(np.float32)
                error_rate, _ = metrics.whisper_cer(wav, sr, text)
                group_results[config_name]["cer"].append(error_rate)
                if tq_config is None:
                    baseline_wav = wav
                    baseline_sr = sr
                elif baseline_wav is not None:
                    spk_sim = metrics.speaker_cosine_similarity(
                        baseline_wav, baseline_sr, wav, sr
                    )
                    group_results[config_name]["spk_sim"].append(spk_sim)

        for config_name, tq_config in TURBOQUANT_CONFIGS:
            r = group_results[config_name]
            avg_cer = sum(r["cer"]) / len(r["cer"]) if r["cer"] else 0
            avg_spk = sum(r["spk_sim"]) / len(r["spk_sim"]) if r["spk_sim"] else 0
            spk_str = f"{avg_spk:.4f}" if tq_config is not None else "---"
            print(f"{config_name:<22} {avg_cer:<10.2%} {spk_str:<12}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark TurboQuant on Qwen3-TTS with real weights"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--device",
        default="cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu"),
        help="Device to run TTS on",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype",
    )
    parser.add_argument(
        "--no-quality",
        action="store_true",
        help="Skip quality metrics (Whisper CER, WavLM similarity)",
    )
    parser.add_argument(
        "--metrics-device",
        default="cpu",
        help="Device for Whisper CER + WavLM speaker-sim (default: cpu). Use 'cuda' "
        "to run them on the GPU — much faster per clip when the GPU is otherwise "
        "idle, and the model has plenty of spare VRAM.",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Skip generation, evaluate saved wavs from a previous run",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run torch.profiler on one sentence per config (skips the 22-sentence sweep)",
    )
    parser.add_argument(
        "--profile-sentence",
        default="long",
        choices=available_groups(),
        help="Which sentence group's first entry to profile under --profile",
    )
    parser.add_argument(
        "--profile-configs",
        default="baseline (no TQ),K4/V2 rw=128",
        help="Comma-separated config names to profile (default: baseline + K4/V2). "
        "Use empty string to profile ALL configs (much slower).",
    )
    parser.add_argument(
        "--groups",
        default=",".join(available_groups()),
        help="Comma-separated sentence groups to run (default: all). Valid: "
        f"{available_groups()}. seedtts_en / ellav_hard need --data-dir "
        "(run fetch-eval-data first). Smoke test: --groups smoke.",
    )
    parser.add_argument(
        "--max-per-group",
        type=int,
        default=None,
        help="Cap sentences per group (useful for smoke tests; default: run all).",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory holding the standard eval sets (en/*.lst + prompt-wavs + "
        "ellav_hard.txt) from tools/fetch_eval_data.py. Required for the "
        "seedtts_en / ellav_hard groups; ignored by the curated groups.",
    )
    parser.add_argument(
        "--voice-mode",
        default="auto",
        choices=["auto", "preset", "clone"],
        help="How to voice each sentence. 'preset' = built-in speaker (needs a "
        "*-CustomVoice checkpoint). 'clone' = zero-shot clone a reference clip "
        "(needs a *-Base checkpoint + --default-ref-audio for groups without one). "
        "'auto' (default) = clone when a reference exists, else preset, falling "
        "back to preset if the checkpoint can't clone.",
    )
    parser.add_argument(
        "--default-ref-audio",
        default=None,
        help="Reference WAV used in clone mode for sentences with no own clip "
        "(the curated groups). Its transcript should be given via --default-ref-text.",
    )
    parser.add_argument(
        "--default-ref-text",
        default=None,
        help="Transcript of --default-ref-audio (for ICL cloning).",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split the work-list across N parallel workers (round-robin over "
        "(group, sentence, seed, temperature) cells). Each shard writes its own "
        "trials CSV. Use WORKERS=1 for the serial performance pass.",
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help="This worker's shard index in [0, num-shards). Only the 0th shard "
        "runs the deterministic intrinsic-distortion probe.",
    )
    parser.add_argument(
        "--run-tag",
        default="",
        help="Shared label embedded in the trials CSV name so the parallel "
        "launcher's shards group into one run for analyze --trials-glob.",
    )
    parser.add_argument(
        "--seeds",
        default="0,1,2,3,4",
        help="Comma-separated generation seeds for repetition (default: 5 seeds). "
        "Each (sentence, seed) reseeds every config identically (paired design). "
        "Smoke test: --seeds 0,1.",
    )
    parser.add_argument(
        "--decode",
        default="both",
        choices=["sampling", "greedy", "both"],
        help="Decode arm(s): 'sampling' (model defaults, stochastic), 'greedy' "
        "(deterministic argmax), or 'both' (default). Greedy isolates compression; "
        "sampling characterises real-world variance.",
    )
    parser.add_argument(
        "--temperatures",
        default="",
        help="Comma-separated sampling temperatures to sweep, e.g. '0.7,1.0,1.2'. "
        "Applies to the SAMPLING arm only (greedy ignores temperature). Empty "
        "(default) = no sweep, use the model's configured temperature. Higher "
        "temperatures flatten the distribution and surface compression instability; "
        "only meaningful together with --track-only-off.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Request deterministic CUDA kernels (torch.use_deterministic_algorithms). "
        "Slower; must run in a fresh process. Use for the greedy reproducibility check.",
    )
    parser.add_argument(
        "--no-intrinsic",
        action="store_true",
        help="Skip the deterministic intrinsic-distortion probe (KV reconstruction).",
    )
    parser.add_argument(
        "--track-only-off",
        action="store_true",
        help="Apply REAL compression during generation (track_only=False) instead "
        "of the default fp16 track-only mode. Required to see compression's genuine "
        "downstream effect on CER/SpkSim — in track-only mode every config produces "
        "identical audio. Decode is substantially slower (legacy on-path compression).",
    )
    args = parser.parse_args()

    # Propagate track_only=False into every TurboQuantConfig if requested, so the
    # generated audio actually reflects compression (not just analytical metrics).
    if args.track_only_off:
        for _, cfg in TURBOQUANT_CONFIGS:
            if cfg is not None:
                cfg.track_only = False

    requested = [g.strip() for g in args.groups.split(",") if g.strip()]
    unknown = [g for g in requested if g not in available_groups()]
    if unknown:
        parser.error(f"unknown group(s): {unknown}. Valid: {available_groups()}")
    args.active_groups = requested

    if args.num_shards < 1 or not (0 <= args.shard_id < args.num_shards):
        parser.error(
            f"invalid sharding: shard-id {args.shard_id} must be in "
            f"[0, num-shards={args.num_shards})"
        )

    try:
        args.seeds = parse_seeds(args.seeds)
    except ValueError as e:
        parser.error(str(e))

    # Empty --temperatures = no sweep (use model default); else parse the list.
    args.temperatures = (
        parse_temperatures(args.temperatures) if args.temperatures.strip() else []
    )

    if args.profile:
        profile_all_configs(args)
    elif args.evaluate_only:
        evaluate_saved_wavs(args)
    else:
        benchmark_qwen3tts(args)


if __name__ == "__main__":
    main()
