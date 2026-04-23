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


# --- Test sentences (varying lengths) ---

SENTENCE_GROUPS = {
    "short": [
        "Hello, how are you doing today?",
        "The weather is beautiful this morning.",
        "Please pass me the salt and pepper.",
        "I will be back in just a minute.",
        "She said she would be here by noon.",
        "Can you help me with this problem?",
        "The train arrives at half past three.",
        "He walked slowly down the empty street.",
        "We need to finish this before Friday.",
        "Thank you very much for your help.",
    ],
    "medium": [
        "The old man sat quietly on the bench, watching the children play in the park while the sun slowly set behind the distant mountains.",
        "Scientists have discovered a new species of deep sea fish that can produce its own light, allowing it to survive in the darkest parts of the ocean.",
        "After years of hard work and dedication, she finally received the promotion she had been hoping for, along with a corner office on the top floor.",
        "The ancient library contained thousands of manuscripts, some dating back over a thousand years, carefully preserved behind glass cases in temperature controlled rooms.",
        "Running a small business requires patience, creativity, and a willingness to adapt to changing market conditions, especially during times of economic uncertainty.",
        "The documentary explored how traditional farming methods are being combined with modern technology to create more sustainable agricultural practices around the world.",
        "Every morning she would walk through the garden, picking fresh herbs for breakfast while listening to the birds sing their familiar songs in the tall oak trees.",
    ],
    "long": [
        (
            "The history of human civilization is a remarkable story of innovation and perseverance. "
            "From the earliest cave paintings to the development of written language, from the invention "
            "of the wheel to the creation of the internet, each generation has built upon the achievements "
            "of those who came before. Today we stand at a crossroads where artificial intelligence and "
            "biotechnology promise to reshape our world in ways we can barely imagine. The choices we make "
            "now will determine the course of human history for centuries to come."
        ),
        (
            "The ocean covers more than seventy percent of the Earth's surface and contains ninety seven "
            "percent of all the water on our planet. Despite centuries of exploration, we have mapped less "
            "than twenty percent of the ocean floor. The deep sea remains one of the last great frontiers, "
            "home to creatures that have evolved in complete darkness, under pressures that would crush "
            "most land animals, and at temperatures near freezing."
        ),
        (
            "Education is the most powerful tool for changing the world. A good education "
            "gives people the ability to think critically, solve complex problems, and communicate "
            "effectively with others. It opens doors to new opportunities and helps break the cycle of poverty. "
            "When we invest in education, we invest in the future of our communities and our nations. Every "
            "child deserves access to quality learning regardless of where they were born or what challenges "
            "they face in their daily lives."
        ),
        (
            "Music has been a fundamental part of human culture for tens of thousands of years. Archaeological "
            "evidence suggests that early humans created simple flutes from bird bones and mammoth ivory over "
            "forty thousand years ago. Throughout history, music has served many purposes, from ceremonies "
            "to entertainment, from communication to emotional expression. Today, music continues "
            "to evolve, blending traditional instruments with digital technology to create entirely new sounds "
            "that would have been unimaginable to our ancestors."
        ),
        (
            "The art of cooking has transformed dramatically over the past century. What was once a daily "
            "necessity focused purely on survival has become a global cultural phenomenon. Chefs travel the "
            "world to study different traditions, combining techniques from Asia, Europe, Africa, and the "
            "Americas to create dishes that tell stories of migration, trade, and human connection. Food "
            "brings people together across languages and borders in ways that few other things can."
        ),
    ],
}

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

    def whisper_cer(self, wav: np.ndarray, sr: int, reference_text: str) -> tuple[float, str]:
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
            self._wavlm_model = WavLMForXVector.from_pretrained(
                "microsoft/wavlm-base-plus-sv"
            ).to(self._device).eval()

    def speaker_embedding(self, wav: np.ndarray, sr: int) -> np.ndarray:
        """Extract normalized speaker embedding (512-dim)."""
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
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            emb = self._wavlm_model(**inputs).embeddings
            emb = torch.nn.functional.normalize(emb, dim=-1)

        return emb.squeeze().cpu().numpy()

    def speaker_cosine_similarity(self, wav_a: np.ndarray, sr_a: int,
                                   wav_b: np.ndarray, sr_b: int) -> float:
        emb_a = self.speaker_embedding(wav_a, sr_a)
        emb_b = self.speaker_embedding(wav_b, sr_b)
        return float(np.dot(emb_a, emb_b))


# ---------------------------------------------------------------------------
# Attention similarity — compress/decompress real K/V and measure cosine sim
# ---------------------------------------------------------------------------

def measure_attention_similarity(model, text, language, speaker):
    """Run baseline generation, then compress/decompress the KV cache to measure
    reconstruction quality across all TurboQuant configs.

    Returns dict: config_name -> avg cosine similarity across all layers.
    """
    from turboquant.compressors_v3 import TurboQuantV3

    # Generate with baseline to get real K/V activations
    wavs, sr = model.generate_custom_voice(text=text, language=language, speaker=speaker)
    cache = model.model.last_kv_cache  # DynamicCache with fp16 K/V

    # Extract K/V from DynamicCache layers
    # transformers >= 4.57 stores K/V in cache.layers[i].key_cache / value_cache
    # older versions use cache.key_cache[i] / cache.value_cache[i]
    if hasattr(cache, "layers") and len(cache.layers) > 0 and hasattr(cache.layers[0], "keys"):
        n_layers = len(cache.layers)
        fp16_keys = [cache.layers[i].keys for i in range(n_layers)]
        fp16_values = [cache.layers[i].values for i in range(n_layers)]
    elif hasattr(cache, "key_cache") and len(cache.key_cache) > 0:
        n_layers = len(cache.key_cache)
        fp16_keys = [cache.key_cache[i] for i in range(n_layers)]
        fp16_values = [cache.value_cache[i] for i in range(n_layers)]
    else:
        print("  ERROR: Cannot extract K/V from cache (unknown format)")
        return {}

    seq_len = fp16_keys[0].shape[2]
    head_dim = fp16_keys[0].shape[3]
    print(f"\n  Attention similarity test ({seq_len} tokens, {n_layers} layers, head_dim={head_dim})")

    configs = [
        ("K4/V2", 4, 2),
        ("K3/V3", 3, 3),
        ("K3/V2", 3, 2),
        ("K2/V2", 2, 2),
    ]

    print(f"  {'Config':<12} {'Key CosSim':<12} {'Val CosSim':<12} {'Avg':<12}")
    print(f"  {'-' * 48}")

    results = {}
    for name, kb, vb in configs:
        key_sims = []
        val_sims = []

        for layer_idx in range(n_layers):
            comp = TurboQuantV3(
                head_dim=head_dim, key_bits=kb, value_bits=vb,
                residual_window=0,  # compress everything for fair comparison
                layer_idx=layer_idx, n_layers=n_layers,
                protected_layers=2, protected_bits=8,
                seed=42, device=str(fp16_keys[layer_idx].device),
            )
            orig_k = fp16_keys[layer_idx]
            orig_v = fp16_values[layer_idx]

            ck, cv = comp.compress_kv(orig_k, orig_v)
            recon_k, recon_v = comp.decompress_kv(ck, cv)

            # Cosine similarity per vector, averaged
            def cos_sim(a, b):
                a_flat = a.reshape(-1, head_dim).float()
                b_flat = b.reshape(-1, head_dim).float()
                cos = torch.nn.functional.cosine_similarity(a_flat, b_flat, dim=-1)
                return cos.mean().item()

            key_sims.append(cos_sim(orig_k, recon_k))
            val_sims.append(cos_sim(orig_v, recon_v))

        avg_k = sum(key_sims) / len(key_sims)
        avg_v = sum(val_sims) / len(val_sims)
        avg = (avg_k + avg_v) / 2
        results[name] = avg

        print(f"  {name:<12} {avg_k:<12.4f} {avg_v:<12.4f} {avg:<12.4f}")

    return results


# ---------------------------------------------------------------------------
# Benchmark logic
# ---------------------------------------------------------------------------

def fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    elif n < 1024 ** 2:
        return f"{n / 1024:.1f} KB"
    else:
        return f"{n / 1024 ** 2:.1f} MB"


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
            f"  total_memory={props.total_memory / (1024 ** 3):.1f} GB",
            f"  torch={torch.__version__} cuda={torch.version.cuda}",
        ]
    out = "\n".join(lines)
    print(out)
    if sink is not None:
        sink.write(out + "\n")
        sink.flush()


def reset_peak_memory(device):
    if _is_cuda(device):
        torch.cuda.reset_peak_memory_stats(device)


def read_peak_memory_mb(device) -> float:
    if not _is_cuda(device):
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


# ---------------------------------------------------------------------------
# Incremental results file (so a crash doesn't lose partial data)
# ---------------------------------------------------------------------------

def _project_root() -> str:
    """Walk up from this file to the repo root (.../TTS-TurboQuant)."""
    # _THIS_DIR = .../models/Qwen3-TTS/benchmarks → 3 dirnames up
    return os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))


def _open_results_file(prefix: str) -> tuple:
    """Open a line-buffered append handle under <repo>/results/. Returns (fh, path)."""
    results_dir = os.path.join(_project_root(), "results")
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(results_dir, f"{prefix}_{ts}.txt")
    fh = open(path, "a", buffering=1, encoding="utf-8")
    fh.write(f"# {prefix} run started {ts}\n")
    fh.write(f"# columns: group,idx,config,rtf,cer,spk_sim,peak_vram_mb,tokens_per_sec,"
             f"sim_compressed_mb,realized_mb,theoretical_ratio,effective_ratio\n")
    return fh, path


def _write_result_line(fh, **kw):
    """Append one CSV-ish row. Missing values render as empty."""
    cols = [
        "group", "idx", "config", "rtf", "cer", "spk_sim",
        "peak_vram_mb", "tokens_per_sec", "sim_compressed_mb",
        "realized_mb", "theoretical_ratio", "effective_ratio",
    ]
    row = []
    for c in cols:
        v = kw.get(c)
        if v is None:
            row.append("")
        elif isinstance(v, float):
            row.append(f"{v:.6g}")
        else:
            row.append(str(v))
    fh.write(",".join(row) + "\n")


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


def run_generation(model, text, language, speaker, tq_config, device=None):
    """Run a single generation.

    Returns (wavs, sr, elapsed, memory_report, peak_vram_mb, n_ar_tokens).
    Wall time covers GPU completion via an explicit sync before/after.
    """
    kwargs = {}
    if tq_config is not None:
        kwargs["turboquant_config"] = tq_config

    if device is None and hasattr(model, "device"):
        device = model.device

    if _is_cuda(device):
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    start = time.time()
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


def benchmark_qwen3tts(args):
    device = args.device

    results_fh, results_path = _open_results_file("benchmark_qwen3tts")
    gpu_csv = start_nvidia_smi_monitor("benchmark_qwen3tts")

    header = [
        "=" * 110,
        "Qwen3-TTS Real-Weights Benchmark",
        f"Model: {args.model}",
        f"Device: {device} | Dtype: {args.dtype} | Quality metrics: {not args.no_quality}",
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
    model = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map=args.device,
        dtype=dtype,
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
        metrics_device = "cpu"
        _tee(results_fh, f"\nLoading quality metrics on {metrics_device}...")
        metrics = QualityMetrics(device=metrics_device)

    # One warmup per config so TurboQuant lazy-init / CUDA kernel autotune
    # doesn't skew the first-sentence measurement.
    _tee(results_fh, "\nWarmup (one per config)...")
    for config_name, tq_config in TURBOQUANT_CONFIGS:
        try:
            run_generation(model, "Hello world.", "English", speaker, tq_config, device=device)
        except Exception as e:
            _tee(results_fh, f"  warmup {config_name}: ERROR {e}")
    _tee(results_fh, "Warmup done.\n")

    output_dir = os.path.join(_THIS_DIR, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    summary = {}
    active_groups = getattr(args, "active_groups", list(SENTENCE_GROUPS.keys()))
    max_per_group = getattr(args, "max_per_group", None)

    for group_name in active_groups:
        texts = SENTENCE_GROUPS[group_name]
        if max_per_group is not None:
            texts = texts[:max_per_group]
        n = len(texts)
        _tee(results_fh, f"\n{'=' * 110}")
        _tee(results_fh, f"Group: {group_name} ({n} sentences)")
        _tee(results_fh, f"{'=' * 110}")

        group_results = {
            name: {
                "rtf": [], "cer": [], "spk_sim": [],
                "theoretical_ratio": [], "effective_ratio": [],
                "peak_vram_mb": [], "tokens_per_sec": [],
                "sim_compressed_mb": [], "realized_mb": [],
            }
            for name, _ in TURBOQUANT_CONFIGS
        }

        for i, text in enumerate(texts):
            text_preview = text[:50] + "..." if len(text) > 50 else text
            _tee(results_fh, f"\n  [{i+1}/{n}] \"{text_preview}\"")

            baseline_wav = None
            baseline_sr = None

            for config_name, tq_config in TURBOQUANT_CONFIGS:
                try:
                    wavs, sr, elapsed, mem_report, peak_vram_mb, n_ar_tokens = run_generation(
                        model, text, "English", speaker, tq_config, device=device,
                    )
                    wav = wavs[0]
                    audio_duration = len(wav) / sr
                    rtf = elapsed / audio_duration if audio_duration > 0 else float("inf")
                    tokens_per_sec = n_ar_tokens / elapsed if elapsed > 0 else 0.0

                    group_results[config_name]["rtf"].append(rtf)
                    group_results[config_name]["peak_vram_mb"].append(peak_vram_mb)
                    group_results[config_name]["tokens_per_sec"].append(tokens_per_sec)

                    theoretical_ratio = 1.0
                    effective_ratio = 1.0
                    sim_compressed_mb = 0.0
                    realized_mb = 0.0
                    if mem_report:
                        theoretical_ratio = mem_report.get("theoretical_compression_ratio",
                                                           mem_report.get("compression_ratio", 1.0))
                        effective_ratio = mem_report.get("effective_compression_ratio", theoretical_ratio)
                        sim_compressed_mb = mem_report.get(
                            "compressed_bytes", 0
                        ) / (1024 ** 2)
                        realized_mb = mem_report.get(
                            "realized_fp16_bytes",
                            mem_report.get("total_bytes", 0),
                        ) / (1024 ** 2)
                        group_results[config_name]["theoretical_ratio"].append(theoretical_ratio)
                        group_results[config_name]["effective_ratio"].append(effective_ratio)
                        group_results[config_name]["sim_compressed_mb"].append(sim_compressed_mb)
                        group_results[config_name]["realized_mb"].append(realized_mb)

                    error_rate = None
                    spk_sim = None
                    if metrics:
                        error_rate, _ = metrics.whisper_cer(wav, sr, text)
                        group_results[config_name]["cer"].append(error_rate)

                        if tq_config is None:
                            baseline_wav = wav
                            baseline_sr = sr
                        elif baseline_wav is not None:
                            spk_sim = metrics.speaker_cosine_similarity(
                                baseline_wav, baseline_sr, wav, sr
                            )
                            if spk_sim is not None:
                                group_results[config_name]["spk_sim"].append(spk_sim)

                    out_path = os.path.join(
                        output_dir,
                        f"qwen_{group_name}_{i}_{config_name.replace(' ', '_').replace('/', '_')}.wav",
                    )
                    sf.write(out_path, wav, sr)

                    status = f"RTF={rtf:.2f} VRAM={peak_vram_mb:.0f}MB tok/s={tokens_per_sec:.1f}"
                    if metrics and error_rate is not None:
                        status += f" CER={error_rate:.1%}"
                        if spk_sim is not None:
                            status += f" SpkSim={spk_sim:.4f}"
                    if mem_report:
                        status += f" Ratio_theory={theoretical_ratio:.2f}x eff={effective_ratio:.2f}x"
                    _tee(results_fh, f"    {config_name:<22} {status}")

                    _write_result_line(
                        results_fh,
                        group=group_name, idx=i, config=config_name,
                        rtf=rtf, cer=error_rate, spk_sim=spk_sim,
                        peak_vram_mb=peak_vram_mb, tokens_per_sec=tokens_per_sec,
                        sim_compressed_mb=sim_compressed_mb,
                        realized_mb=realized_mb,
                        theoretical_ratio=theoretical_ratio,
                        effective_ratio=effective_ratio,
                    )

                except Exception as e:
                    _tee(results_fh, f"    {config_name:<22} ERROR: {e}")
                    import traceback
                    traceback.print_exc()

        # Per-group averages
        _tee(results_fh, f"\n  {'─' * 80}")
        _tee(results_fh, f"  AVERAGES for {group_name} ({n} sentences):")
        _tee(results_fh, f"  {'─' * 80}")
        if metrics:
            _tee(results_fh, f"  {'Config':<22} {'RTF':<7} {'CER':<8} {'SpkSim':<8} "
                             f"{'VRAM(MB)':<10} {'tok/s':<8} {'R_th':<6} {'R_eff':<6}")
        else:
            _tee(results_fh, f"  {'Config':<22} {'RTF':<7} {'VRAM(MB)':<10} {'tok/s':<8} "
                             f"{'R_th':<6} {'R_eff':<6}")
        _tee(results_fh, f"  {'-' * 90}")

        def _avg(xs, default=0.0):
            return sum(xs) / len(xs) if xs else default

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
                _tee(results_fh, f"  {config_name:<22} {avg_rtf:<7.2f} {avg_cer:<8.2%} "
                                 f"{spk_str:<8} {avg_vram:<10.0f} {avg_tps:<8.1f} "
                                 f"{avg_th:<6.2f} {avg_eff:<6.2f}")
            else:
                _tee(results_fh, f"  {config_name:<22} {avg_rtf:<7.2f} {avg_vram:<10.0f} "
                                 f"{avg_tps:<8.1f} {avg_th:<6.2f} {avg_eff:<6.2f}")

        summary[group_name] = group_results

    # ----- FINAL SUMMARY: Quality -----
    if metrics:
        _tee(results_fh, f"\n{'=' * 110}")
        _tee(results_fh, "FINAL SUMMARY — Quality (averages across all sentences)")
        _tee(results_fh, f"{'=' * 110}")
        header_row = f"{'Config':<22} "
        for group_name in active_groups:
            header_row += f"{'RTF':<7} {'CER':<7} {'SpkSim':<9} "
        _tee(results_fh, header_row)
        label_row = f"{'':22} "
        for group_name in active_groups:
            n = len(SENTENCE_GROUPS[group_name])
            label_row += f"{group_name + f' ({n})':<24}"
        _tee(results_fh, label_row)

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

    # ----- FINAL SUMMARY: Memory & Throughput -----
    _tee(results_fh, f"\n{'=' * 110}")
    _tee(results_fh, "FINAL SUMMARY — Memory & Throughput (averages across all sentences)")
    _tee(results_fh, f"{'=' * 110}")
    _tee(results_fh, f"{'Config':<22} {'VRAM(MB)':<10} {'tok/s':<8} {'Realized(MB)':<14} "
                     f"{'SimComp(MB)':<14} {'R_theory':<10} {'R_eff':<8}")
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
        _tee(results_fh,
             f"{config_name:<22} {avg_vram:<10.0f} {avg_tps:<8.1f} "
             f"{avg_realized:<14.2f} {avg_sim:<14.2f} {avg_th:<10.2f} {avg_eff:<8.2f}")

    _tee(results_fh, f"\nOutput audio saved to: {output_dir}/")
    _tee(results_fh, f"Structured results: {results_path}")

    # Attention similarity test — averaged per group
    _tee(results_fh, f"\n{'=' * 110}")
    _tee(results_fh, "KV Cache Reconstruction Quality (attention similarity)")
    _tee(results_fh, f"{'=' * 110}")

    configs = [("K4/V2", 4, 2), ("K3/V3", 3, 3), ("K3/V2", 3, 2), ("K2/V2", 2, 2)]
    all_group_avgs = {}

    for group_name in active_groups:
        texts = SENTENCE_GROUPS[group_name]
        if max_per_group is not None:
            texts = texts[:max_per_group]
        n = len(texts)
        _tee(results_fh, f"\n  [{group_name}] ({n} sentences)")
        group_sims = {name: [] for name, _, _ in configs}

        for i, text in enumerate(texts):
            result = measure_attention_similarity(model, text, "English", speaker)
            for name in result:
                group_sims[name].append(result[name])

        _tee(results_fh, f"\n  {'Config':<12} {'Avg Similarity':<16} {'Min':<10} {'Max':<10}")
        _tee(results_fh, f"  {'-' * 48}")
        for name, _, _ in configs:
            vals = group_sims[name]
            if vals:
                avg = sum(vals) / len(vals)
                _tee(results_fh, f"  {name:<12} {avg:<16.4f} {min(vals):<10.4f} {max(vals):<10.4f}")
                all_group_avgs.setdefault(name, {})[group_name] = avg

    _tee(results_fh, f"\n  {'─' * 60}")
    _tee(results_fh, f"  Attention Similarity Summary:")
    header = f"  {'Config':<12}"
    for group_name in active_groups:
        header += f" {group_name:<12}"
    _tee(results_fh, header)
    for name, _, _ in configs:
        row = f"  {name:<12}"
        for group_name in active_groups:
            val = all_group_avgs.get(name, {}).get(group_name, 0)
            row += f" {val:<12.4f}"
        _tee(results_fh, row)

    results_fh.close()


# ---------------------------------------------------------------------------
# Profiling mode (mirrors VALL-E benchmark)
# ---------------------------------------------------------------------------

@torch.no_grad()
def profile_generation(model, text, language, speaker, tq_config, device,
                       output_dir, config_name, group_name):
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
    with profile(activities=activities,
                 record_shapes=False,
                 profile_memory=False,
                 with_stack=False) as prof:
        model.generate_custom_voice(
            text=text, language=language, speaker=speaker, **kwargs,
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
        label = (f"LAUNCH-BOUND (cuda_time/wall = {frac:.1%}): batching would likely help. "
                 f"Inspect Chrome trace for the dominant launch cost.")
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

    text = SENTENCE_GROUPS[args.profile_sentence][0]
    _tee(results_fh, f"Profiling text ({args.profile_sentence}): \"{text[:80]}...\"")

    output_dir = os.path.join(_THIS_DIR, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Filter which configs to profile (default: baseline + K4/V2).
    requested = getattr(args, "profile_configs", None)
    if requested:
        names = [c.strip() for c in requested.split(",") if c.strip()]
        configs_to_run = [(n, c) for (n, c) in TURBOQUANT_CONFIGS if n in names]
        if not configs_to_run:
            _tee(results_fh, f"ERROR: --profile-configs matched nothing. "
                             f"Requested: {names}. Available: {[n for n, _ in TURBOQUANT_CONFIGS]}")
            results_fh.close()
            return
    else:
        configs_to_run = TURBOQUANT_CONFIGS

    # Per-config warmup
    _tee(results_fh, "\nWarmup (one per config)...")
    for config_name, tq_config in configs_to_run:
        try:
            run_generation(model, "Hello world.", "English", speaker, tq_config, device=device)
        except Exception as e:
            _tee(results_fh, f"  warmup {config_name}: ERROR {e}")
    _tee(results_fh, "Warmup done.\n")

    for config_name, tq_config in configs_to_run:
        _tee(results_fh, f"\n{'=' * 110}")
        _tee(results_fh, f"Profiling config: {config_name}")
        _tee(results_fh, f"{'=' * 110}")
        try:
            prof, elapsed, trace_path = profile_generation(
                model, text, "English", speaker, tq_config, device,
                output_dir, config_name, args.profile_sentence,
            )

            peak_vram_mb = read_peak_memory_mb(device)
            mem_report = None
            if hasattr(model.model, "last_kv_cache") and model.model.last_kv_cache is not None:
                cache = model.model.last_kv_cache
                if hasattr(cache, "memory_report"):
                    mem_report = cache.memory_report()

            _tee(results_fh, f"Elapsed: {elapsed:.2f}s  Peak VRAM: {peak_vram_mb:.0f} MB")
            if mem_report:
                comp_mb = mem_report.get("compressed_bytes", 0) / 1024 ** 2
                real_mb = mem_report.get("realized_fp16_bytes",
                                         mem_report.get("total_bytes", 0)) / 1024 ** 2
                _tee(results_fh,
                     f"Realized: {real_mb:.2f} MB | SimComp: {comp_mb:.2f} MB | "
                     f"R_theory={mem_report.get('theoretical_compression_ratio', mem_report.get('compression_ratio', 1.0)):.2f}x "
                     f"R_eff={mem_report.get('effective_compression_ratio', 1.0):.2f}x")

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
                cpu_table = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10)
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
        print(f"ERROR: No outputs found at {output_dir}. Run generation first (without --evaluate-only).")
        return

    print("=" * 80)
    print("Qwen3-TTS Quality Evaluation (from saved wavs)")
    print("=" * 80)

    metrics = QualityMetrics(device="cpu")

    for group_name, texts in SENTENCE_GROUPS.items():
        print(f"\n{'─' * 80}")
        print(f"Group: {group_name} ({len(texts)} sentences)")
        print(f"{'─' * 80}")
        print(f"{'Config':<22} {'Avg CER':<10} {'Avg SpkSim':<12}")
        print("-" * 45)

        group_results = {name: {"cer": [], "spk_sim": []} for name, _ in TURBOQUANT_CONFIGS}

        for i, text in enumerate(texts):
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
                    spk_sim = metrics.speaker_cosine_similarity(baseline_wav, baseline_sr, wav, sr)
                    group_results[config_name]["spk_sim"].append(spk_sim)

        for config_name, tq_config in TURBOQUANT_CONFIGS:
            r = group_results[config_name]
            avg_cer = sum(r["cer"]) / len(r["cer"]) if r["cer"] else 0
            avg_spk = sum(r["spk_sim"]) / len(r["spk_sim"]) if r["spk_sim"] else 0
            spk_str = f"{avg_spk:.4f}" if tq_config is not None else "---"
            print(f"{config_name:<22} {avg_cer:<10.2%} {spk_str:<12}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark TurboQuant on Qwen3-TTS with real weights")
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                        help="HuggingFace model name or local path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else
                        ("mps" if torch.backends.mps.is_available() else "cpu"),
                        help="Device to run TTS on")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"],
                        help="Model dtype")
    parser.add_argument("--no-quality", action="store_true",
                        help="Skip quality metrics (Whisper CER, WavLM similarity)")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Skip generation, evaluate saved wavs from a previous run")
    parser.add_argument("--profile", action="store_true",
                        help="Run torch.profiler on one sentence per config (skips the 22-sentence sweep)")
    parser.add_argument("--profile-sentence", default="medium",
                        choices=list(SENTENCE_GROUPS.keys()),
                        help="Which sentence group's first entry to profile under --profile")
    parser.add_argument("--profile-configs",
                        default="baseline (no TQ),K4/V2 rw=128",
                        help="Comma-separated config names to profile (default: baseline + K4/V2). "
                             "Use empty string to profile ALL configs (much slower).")
    parser.add_argument("--groups", default=",".join(SENTENCE_GROUPS.keys()),
                        help="Comma-separated sentence groups to run (default: all). "
                             "Useful on tight-VRAM GPUs: --groups short,medium to skip long sentences. "
                             "For a smoke test: --groups short.")
    parser.add_argument("--max-per-group", type=int, default=None,
                        help="Cap sentences per group (useful for smoke tests; default: run all).")
    args = parser.parse_args()

    requested = [g.strip() for g in args.groups.split(",") if g.strip()]
    unknown = [g for g in requested if g not in SENTENCE_GROUPS]
    if unknown:
        parser.error(f"unknown group(s): {unknown}. Valid: {list(SENTENCE_GROUPS)}")
    args.active_groups = requested

    if args.profile:
        profile_all_configs(args)
    elif args.evaluate_only:
        evaluate_saved_wavs(args)
    else:
        benchmark_qwen3tts(args)


if __name__ == "__main__":
    main()
