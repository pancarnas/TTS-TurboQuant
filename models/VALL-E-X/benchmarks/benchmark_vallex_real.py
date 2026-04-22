"""Benchmark TurboQuant KV cache on VALL-E-X with real model weights.

Loads the actual VALL-E-X checkpoint, generates speech with and without
TurboQuant compression, and compares:
  - Latency (RTF)
  - Whisper CER (character error rate)
  - WavLM speaker cosine similarity vs baseline

Mirrors models/Qwen3-TTS/benchmarks/benchmark_qwen3tts_real.py in structure
(same 22 English sentences, same 5 TurboQuant configs) so results are directly
comparable to the Qwen3-TTS table in the README.

Requires:
    - VALL-E-X checkpoint (auto-downloaded on first run to models/VALL-E-X/checkpoints/)
    - pip install -e models/VALL-E-X/  # installs benchmark deps
    - pip install -e .                  # installs turboquant

Usage:
    python models/VALL-E-X/benchmarks/benchmark_vallex_real.py \
        [--device cuda] [--no-quality] [--evaluate-only] \
        [--profile] [--profile-sentence {short,medium,long}]

Outputs:
    - Per-sentence wavs under models/VALL-E-X/benchmarks/outputs/
    - Incremental structured results at
      results/benchmark_vallex_<timestamp>_results.txt (survives crashes)
    - Chrome profiler traces (when --profile) at
      models/VALL-E-X/benchmarks/outputs/profile_<group>_<config>.json.gz

L4 findings and phased rework (2026-04-21):

Phase-0 smoke (pre-fix):
    baseline: RTF 0.58, 130 tok/s, ~2 GB VRAM
    K4/V2:    RTF 4.03, 18.6 tok/s, ~2.7 GB VRAM, eff_ratio 0.78x
    Root cause (profile): decode is launch-bound (7% GPU util, 138k kernels in
    baseline). TurboQuant added 60k+ torch.cat(), 65k aten::to, 13k
    cudaStreamSynchronize events in turboquant_cache.update() by rebuilding
    full K/V from compressed + residual + decompressed-prefix each step.

Phase 1 (commit 2c4c6d7): preallocated per-layer fp16 buffers in
    TurboQuantValleCache. Slice-write replaces cat-per-step; compression moves
    off-path (config.track_only=True, default). Expected: ~0.27 ms/step
    amortized, TQ RTF within 1.2x of baseline, aten::cat count at baseline
    level. effective_compression_ratio intentionally reports 1.0 (honest —
    we store fp16 now, not compressed).

Phase 2 (commit 72c25bb): dtype churn in MSECompressor.compress/decompress
    cleaned up — int32 throughout bit-packing path, removes the
    int64<->uint8 ping-pong. Only matters for --track-only-off A/B mode and
    future compression-aware attention work.

Phase 3: decision gate on L4 profile data.
    If Phase 1/2 latency lands within ~1.2x of baseline → move to Phase 4
    decision. If not → diagnose residual overhead before anything else.

Phase 4 design sketch — realizing REAL memory savings (eff_ratio > 1):
    The fundamental constraint: VALL-E-X's attention at modules/activation.py
    lines 172-175 is standard PyTorch manual softmax:
        att = (q @ k.transpose(-2, -1)) * scale
        att = att.masked_fill(mask, -inf); att = F.softmax(att, dim=-1)
        y = att @ v
    It requires the full (B, nh, FULL_T, D) K/V tensors materialized on the
    GPU. As long as attention is this shape, no amount of cache-side
    compression can reduce real VRAM — decompressing just to feed attention
    defeats the point.

    Four realistic paths, in increasing cost:

    4a) DECLARE AND STOP. Phase 1/2 delivers a correct instrumented baseline
        that proves MSE-optimal quantization's theoretical compression ratio
        (2.5-3x at K4/V2). Real savings are documented as gated on an
        attention-kernel rewrite. Cost: 0. Outcome: honest paper/report
        without the memory-savings claim.

    4b) LAYER-ADAPTIVE DROP. Keep first K and last K layers in fp16
        (semantically important), drop old tokens entirely from middle layers
        beyond a sliding window. Real memory savings ~= window/full_len for
        dropped layers. Cost: ~1 week. Risk: quality regression on long
        sentences; speaker-prompt tokens must never be dropped.

    4c) JIT DECOMPRESSION IN PYTHON. Replace the standard attention call
        with a Python loop: for each chunk of compressed K/V, decompress into
        a temporary fp16 tile, compute q @ k_tile, accumulate via online
        softmax (LogSumExp), then q @ v_tile weighted-summed. Cost: ~2
        weeks. Expected perf: adds many kernel launches — may regress
        latency on launch-bound decode despite Phase 1's fixes. Memory:
        real savings bounded by tile size.

    4d) CUSTOM TRITON KERNEL. Fused dequantize + flash-style tiled attention
        in a single kernel. Loads compressed K/V chunks into shared memory,
        dequantizes in registers, runs online softmax. Cost: 4-8 weeks.
        Expected: real memory savings AND latency parity with baseline.
        Highest upside, highest risk.

    Recommendation depends on L4 numbers:
      - If Phase 1/2 RTF ≈ 0.7 (close to baseline): go 4a (declare). The
        instrumented baseline is the contribution.
      - If the research investment is available: 4d is the only path that
        realizes compression in production. 4c is a dead end given the
        launch-bound baseline.
      - 4b is a hybrid that partially realizes compression at modest cost,
        but only if dropping middle-layer context is acceptable for TTS
        quality (needs validation — VALL-E-X may depend heavily on
        cross-layer context for speaker consistency).
"""

import sys
import os
import time
import argparse
import atexit
import datetime
import shutil
import subprocess

# VALL-E-X uses bare module imports (e.g. `from models.vallex import VALLE`);
# add its directory to sys.path so those resolve.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_VALLEX_DIR = os.path.dirname(_THIS_DIR)  # models/VALL-E-X
sys.path.insert(0, _VALLEX_DIR)

import pathlib
import platform
if platform.system().lower() != "windows":
    pathlib.WindowsPath = pathlib.PosixPath

import numpy as np
import torch
import soundfile as sf
import librosa

from models.vallex import VALLE
from macros import (
    N_DIM, NUM_HEAD, NUM_LAYERS, PREFIX_MODE, NUM_QUANTIZERS,
    SAMPLE_RATE, lang2token,
)
from data.tokenizer import AudioTokenizer
from data.collation import get_text_token_collater
from utils.g2p import PhonemeBpeTokenizer
from turboquant_cache import TurboQuantConfig


# --- Test sentences (mirrors Qwen3-TTS benchmark for direct A/B comparison) ---

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
# Quality metrics (mirrors Qwen benchmark)
# ---------------------------------------------------------------------------

class QualityMetrics:
    """Lazy-loaded quality evaluation: Whisper CER, WavLM speaker sim."""

    def __init__(self, device="cpu"):
        self._device = device
        self._whisper = None
        self._wavlm_model = None
        self._wavlm_extractor = None

    def _load_whisper(self):
        if self._whisper is None:
            import whisper
            self._whisper = whisper.load_model("base", device=self._device)

    def whisper_cer(self, wav: np.ndarray, sr: int, reference_text: str) -> tuple[float, str]:
        self._load_whisper()
        from jiwer import cer
        wav = wav.astype(np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        result = self._whisper.transcribe(wav)
        transcript = result["text"].strip()
        ref = reference_text.strip()
        if not ref:
            return 0.0, transcript
        return float(cer(ref, transcript)), transcript

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
        self._load_wavlm()
        wav = wav.astype(np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
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
# Model loading & generation
# ---------------------------------------------------------------------------

def fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 ** 2:
        return f"{n / 1024:.1f} KB"
    return f"{n / 1024 ** 2:.1f} MB"


# ---------------------------------------------------------------------------
# GPU + memory instrumentation
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
# External nvidia-smi poller — survives Python OOM crashes so the last
# memory reading before a crash is preserved on disk.
# ---------------------------------------------------------------------------

_MONITOR_PROC = None
_MONITOR_FH = None


def _stop_nvidia_smi_monitor():
    """atexit hook: stop the poller and close its file."""
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

    Writes a CSV with (timestamp, memory.used, memory.total, utilization.gpu,
    temperature.gpu) at `interval_s` to results/<prefix>_gpu_<ts>.csv.

    Returns the CSV path, or empty string if nvidia-smi isn't available /
    no CUDA is present.
    """
    global _MONITOR_PROC, _MONITOR_FH

    if shutil.which("nvidia-smi") is None:
        return ""
    if not torch.cuda.is_available():
        return ""

    repo_root = os.path.dirname(os.path.dirname(_VALLEX_DIR))
    results_dir = os.path.join(repo_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(results_dir, f"{prefix}_gpu_{ts}.csv")

    # nvidia-smi's -lms flag accepts a millisecond interval; -l accepts seconds.
    # Using -lms to support sub-second polling if needed.
    interval_ms = max(int(interval_s * 1000), 100)
    cmd = [
        "nvidia-smi",
        "--query-gpu=timestamp,memory.used,memory.total,utilization.gpu,temperature.gpu",
        "--format=csv,nounits",
        f"-lms={interval_ms}",
    ]

    try:
        fh = open(csv_path, "w", buffering=1, encoding="utf-8")
        # Flush header synchronously, then let nvidia-smi stream rows.
        fh.write(f"# gpu monitor started {ts}, interval={interval_ms}ms\n")
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"WARNING: could not start nvidia-smi monitor: {e}")
        return ""

    _MONITOR_PROC = proc
    _MONITOR_FH = fh
    atexit.register(_stop_nvidia_smi_monitor)
    return csv_path


# ---------------------------------------------------------------------------
# Incremental results file (so a crash doesn't lose partial data)
# ---------------------------------------------------------------------------

def _open_results_file(prefix: str) -> tuple:
    """Open a line-buffered append handle under <repo>/results/.

    Returns (handle, path). Caller is responsible for closing.
    """
    repo_root = os.path.dirname(os.path.dirname(_VALLEX_DIR))  # .../TTS-TurboQuant
    results_dir = os.path.join(repo_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(results_dir, f"{prefix}_{ts}.txt")
    # buffering=1 → line-buffered: each line flushes on newline
    fh = open(path, "a", buffering=1, encoding="utf-8")
    fh.write(f"# {prefix} run started {ts}\n")
    fh.write(f"# columns: group,idx,config,rtf,cer,spk_sim,peak_vram_mb,tokens_per_sec,"
             f"sim_compressed_mb,realized_mb,theoretical_ratio,effective_ratio\n")
    return fh, path


def _write_result_line(fh, **kw):
    """Append one CSV-ish row. Missing values render as empty.

    sim_compressed_mb: analytical — what compression WOULD store at configured bits.
    realized_mb: actual VRAM used by the cache's fp16 buffer (track_only=True default).
    theoretical_ratio: fp16_equiv / sim_compressed (how effective compression would be).
    effective_ratio: fp16_equiv / realized (what we actually save; 1.0 in track_only mode).
    """
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


def load_vallex_model(device):
    checkpoints_dir = os.path.join(_VALLEX_DIR, "checkpoints")
    checkpoint_path = os.path.join(checkpoints_dir, "vallex-checkpoint.pt")

    os.makedirs(checkpoints_dir, exist_ok=True)

    if not os.path.exists(checkpoint_path):
        print("Downloading VALL-E-X checkpoint from HuggingFace...")
        import wget
        wget.download(
            "https://huggingface.co/Plachta/VALL-E-X/resolve/main/vallex-checkpoint.pt",
            out=checkpoint_path,
            bar=wget.bar_adaptive,
        )
        print()

    model = VALLE(
        N_DIM, NUM_HEAD, NUM_LAYERS,
        norm_first=True, add_prenet=False, prefix_mode=PREFIX_MODE,
        share_embedding=True, nar_scale_factor=1.0,
        prepend_bos=True, num_quantizers=NUM_QUANTIZERS,
    ).half().to(device)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()

    codec = AudioTokenizer(device)

    try:
        from vocos import Vocos
        vocos = Vocos.from_pretrained("charactr/vocos-encodec-24khz").to(device)
    except ImportError:
        vocos = None
        print("Warning: vocos not installed, will use encodec decoder")

    return model, codec, vocos


def decode_audio(codec, vocos, codes, device):
    if vocos is not None:
        frames = codes.permute(2, 0, 1)
        features = vocos.codes_to_features(frames).float()
        samples = vocos.decode(features, bandwidth_id=torch.tensor([2], device=device))
        return samples.squeeze().cpu().numpy()
    return codec.decode([(codes, None)]).squeeze().cpu().numpy()


@torch.no_grad()
def run_generation(model, codec, vocos, text, language, preset_path, device, tq_config):
    """Run one generation end-to-end.

    Returns (wav, elapsed, memory_report, peak_vram_mb, n_ar_tokens).
    Wall time covers GPU completion via an explicit sync before/after inference.
    """
    text_tokenizer = PhonemeBpeTokenizer(
        tokenizer_path=os.path.join(_VALLEX_DIR, "utils", "g2p", "bpe_69.json")
    )
    text_collater = get_text_token_collater()

    prompt_data = np.load(preset_path)
    audio_prompts = torch.tensor(prompt_data["audio_tokens"]).int().to(device)
    text_prompts = torch.tensor(prompt_data["text_tokens"]).int()
    lang_pr = {0: "zh", 1: "ja", 2: "en"}[int(prompt_data["lang_code"])]
    enroll_x_lens = text_prompts.shape[-1]

    lang_token = lang2token[language]
    formatted_text = lang_token + text + lang_token

    phone_tokens, langs = text_tokenizer.tokenize(text=f"_{formatted_text}".strip())
    text_tokens, text_tokens_lens = text_collater([phone_tokens])
    text_tokens = torch.cat([text_prompts, text_tokens], dim=-1)
    text_tokens_lens += enroll_x_lens

    if _is_cuda(device):
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    start = time.time()
    encoded_frames = model.inference(
        text_tokens.to(device),
        text_tokens_lens.to(device),
        audio_prompts,
        enroll_x_lens=enroll_x_lens,
        top_k=-100,
        temperature=1.0,
        prompt_language=lang_pr,
        text_language=langs,
        turboquant_config=tq_config,
    )
    if _is_cuda(device):
        torch.cuda.synchronize(device)
    elapsed = time.time() - start

    peak_vram_mb = read_peak_memory_mb(device)
    # encoded_frames shape (B=1, T, num_quantizers) — T is the AR-decoded token count.
    n_ar_tokens = int(encoded_frames.shape[1]) if encoded_frames.ndim >= 2 else 0

    wav = decode_audio(codec, vocos, encoded_frames, device)

    memory_report = None
    if tq_config is not None and hasattr(model, "_tq_cache_manager"):
        memory_report = model._tq_cache_manager.memory_report()

    return wav, elapsed, memory_report, peak_vram_mb, n_ar_tokens


# ---------------------------------------------------------------------------
# Main benchmark loop — mirrors Qwen's per-group averaging structure
# ---------------------------------------------------------------------------

def _out_path(output_dir: str, group_name: str, idx: int, config_name: str) -> str:
    safe = config_name.replace(" ", "_").replace("/", "_")
    return os.path.join(output_dir, f"vallex_{group_name}_{idx}_{safe}.wav")


def _tee(fh, msg):
    """Print to stdout and to the results file (if any)."""
    print(msg)
    if fh is not None:
        fh.write(msg + "\n")


def benchmark_vallex(args):
    device = torch.device(args.device)

    results_fh, results_path = _open_results_file("benchmark_vallex")
    gpu_csv = start_nvidia_smi_monitor("benchmark_vallex")

    header = [
        "=" * 110,
        "VALL-E-X Real-Weights Benchmark",
        f"Device: {device} | Quality metrics: {not args.no_quality}",
        f"Structured results: {results_path}",
        f"GPU monitor CSV: {gpu_csv or '(nvidia-smi unavailable — skipped)'}",
        "=" * 110,
    ]
    for line in header:
        _tee(results_fh, line)

    log_gpu_config(device, sink=results_fh)

    _tee(results_fh, "\nLoading VALL-E-X model...")
    t0 = time.time()
    model, codec, vocos = load_vallex_model(device)
    _tee(results_fh, f"Model loaded in {time.time() - t0:.1f}s")

    preset_path = os.path.join(_VALLEX_DIR, "presets", args.preset)
    if not os.path.exists(preset_path):
        _tee(results_fh, f"ERROR: preset {args.preset} not found at {preset_path}")
        results_fh.close()
        return
    _tee(results_fh, f"Using preset: {os.path.basename(preset_path)}")

    metrics = None
    if not args.no_quality:
        _tee(results_fh, f"\nLoading quality metrics on {args.metrics_device}...")
        metrics = QualityMetrics(device=args.metrics_device)

    # One warmup per config so TurboQuant lazy-init / CUDA kernel autotune
    # doesn't skew the first-sentence measurement.
    _tee(results_fh, "\nWarmup (one per config)...")
    for config_name, tq_config in TURBOQUANT_CONFIGS:
        try:
            run_generation(model, codec, vocos, "Hello.", "en", preset_path, device, tq_config)
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
            preview = text[:50] + "..." if len(text) > 50 else text
            _tee(results_fh, f"\n  [{i + 1}/{n}] \"{preview}\"")

            baseline_wav = None

            for config_name, tq_config in TURBOQUANT_CONFIGS:
                try:
                    wav, elapsed, mem_report, peak_vram_mb, n_ar_tokens = run_generation(
                        model, codec, vocos, text, "en", preset_path, device, tq_config,
                    )
                    audio_duration = len(wav) / SAMPLE_RATE
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
                        # Simulated: what compression WOULD use (analytical); Realized: what the fp16 buffer actually uses.
                        sim_compressed_mb = mem_report.get(
                            "simulated_compressed_bytes",
                            mem_report.get("compressed_bytes", 0),
                        ) / (1024 ** 2)
                        realized_mb = mem_report.get(
                            "realized_fp16_bytes",
                            mem_report.get("total_bytes", 0),
                        ) / (1024 ** 2)
                        group_results[config_name]["theoretical_ratio"].append(theoretical_ratio)
                        group_results[config_name]["effective_ratio"].append(effective_ratio)
                        group_results[config_name]["sim_compressed_mb"].append(sim_compressed_mb)
                        group_results[config_name]["realized_mb"].append(realized_mb)

                    status = f"RTF={rtf:.2f} VRAM={peak_vram_mb:.0f}MB tok/s={tokens_per_sec:.1f}"
                    error_rate = None
                    spk_sim = None

                    if metrics:
                        error_rate, _ = metrics.whisper_cer(wav, SAMPLE_RATE, text)
                        group_results[config_name]["cer"].append(error_rate)

                        if tq_config is None:
                            baseline_wav = wav
                        elif baseline_wav is not None:
                            spk_sim = metrics.speaker_cosine_similarity(
                                baseline_wav, SAMPLE_RATE, wav, SAMPLE_RATE
                            )
                            group_results[config_name]["spk_sim"].append(spk_sim)

                        status += f" CER={error_rate:.1%}"
                        if spk_sim is not None:
                            status += f" SpkSim={spk_sim:.4f}"

                    if mem_report:
                        status += f" Ratio_theory={theoretical_ratio:.2f}x eff={effective_ratio:.2f}x"

                    sf.write(_out_path(output_dir, group_name, i, config_name), wav, SAMPLE_RATE)
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

    # ----- FINAL SUMMARY -----
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

    # Memory & throughput summary — across ALL sentences per config.
    # Columns explained:
    #   VRAM(MB)      — peak torch allocator VRAM during the step (includes model + activations + cache)
    #   Realized(MB)  — what the KV cache itself actually uses (fp16 buffer in track_only mode)
    #   SimComp(MB)   — analytical: what compression WOULD use if realized (not actual storage)
    #   R_theory      — fp16_equivalent / SimComp, i.e. the compression ratio if it were realized
    #   R_eff         — fp16_equivalent / Realized, i.e. actual memory savings (≈1.0 in track_only)
    _tee(results_fh, f"\n{'=' * 110}")
    _tee(results_fh, "FINAL SUMMARY — Memory & Throughput (averages across all sentences)")
    _tee(results_fh, f"{'=' * 110}")
    _tee(results_fh, f"{'Config':<22} {'VRAM(MB)':<10} {'tok/s':<8} {'Realized(MB)':<14} "
                     f"{'SimComp(MB)':<14} {'R_theory':<10} {'R_eff':<8}")
    _tee(results_fh, "-" * 100)

    for config_name, tq_config in TURBOQUANT_CONFIGS:
        vram = []
        tps = []
        realized = []
        sim_comp = []
        th = []
        eff = []
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
        avg_sim_comp = sum(sim_comp) / len(sim_comp) if sim_comp else 0
        avg_th = sum(th) / len(th) if th else (1.0 if tq_config is None else 0)
        avg_eff = sum(eff) / len(eff) if eff else (1.0 if tq_config is None else 0)
        _tee(results_fh,
             f"{config_name:<22} {avg_vram:<10.0f} {avg_tps:<8.1f} "
             f"{avg_realized:<14.2f} {avg_sim_comp:<14.2f} {avg_th:<10.2f} {avg_eff:<8.2f}")

    _tee(results_fh, f"\nOutput audio saved to: {output_dir}/")
    _tee(results_fh, f"Structured results: {results_path}")
    results_fh.close()


# ---------------------------------------------------------------------------
# Profiling mode
# ---------------------------------------------------------------------------

@torch.no_grad()
def profile_generation(model, codec, vocos, text, preset_path, device,
                       tq_config, output_dir, config_name, group_name):
    """Profile a single inference with torch.profiler. Returns (prof, elapsed, trace_path)."""
    from torch.profiler import profile, ProfilerActivity

    text_tokenizer = PhonemeBpeTokenizer(
        tokenizer_path=os.path.join(_VALLEX_DIR, "utils", "g2p", "bpe_69.json")
    )
    text_collater = get_text_token_collater()
    prompt_data = np.load(preset_path)
    audio_prompts = torch.tensor(prompt_data["audio_tokens"]).int().to(device)
    text_prompts = torch.tensor(prompt_data["text_tokens"]).int()
    lang_pr = {0: "zh", 1: "ja", 2: "en"}[int(prompt_data["lang_code"])]
    enroll_x_lens = text_prompts.shape[-1]
    formatted_text = lang2token["en"] + text + lang2token["en"]
    phone_tokens, langs = text_tokenizer.tokenize(text=f"_{formatted_text}".strip())
    text_tokens, text_tokens_lens = text_collater([phone_tokens])
    text_tokens = torch.cat([text_prompts, text_tokens], dim=-1)
    text_tokens_lens += enroll_x_lens

    activities = [ProfilerActivity.CPU]
    if _is_cuda(device):
        activities.append(ProfilerActivity.CUDA)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    start = time.time()
    # Lightweight profile: record_shapes + profile_memory generate huge traces on
    # autoregressive decode (500k+ events under TurboQuant), and export_chrome_trace
    # then takes hours. Kernel timing alone is enough to identify the bottleneck.
    with profile(activities=activities,
                 record_shapes=False,
                 profile_memory=False,
                 with_stack=False) as prof:
        model.inference(
            text_tokens.to(device),
            text_tokens_lens.to(device),
            audio_prompts,
            enroll_x_lens=enroll_x_lens,
            top_k=-100,
            temperature=1.0,
            prompt_language=lang_pr,
            text_language=langs,
            turboquant_config=tq_config,
        )
    if _is_cuda(device):
        torch.cuda.synchronize(device)
    elapsed = time.time() - start

    safe = config_name.replace(" ", "_").replace("/", "_")
    trace_path = os.path.join(output_dir, f"profile_{group_name}_{safe}.json.gz")
    prof.export_chrome_trace(trace_path)
    return prof, elapsed, trace_path


def _profile_ratio_classification(prof, elapsed_s: float) -> tuple[float, str]:
    """Return (cuda_time_fraction, label). Label answers: GPU-bound vs launch-bound?

    If CUDA kernels occupy most of the wall time → GPU-bound: batching won't help.
    If much of the wall time is outside CUDA kernels → launch-bound: batching likely helps.
    """
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
                 f"Follow-up: patch vallex.py:492 for batch>1.")
    return frac, label


def profile_all_configs(args):
    """Profile one representative sentence per TurboQuant config."""
    device = torch.device(args.device)
    results_fh, results_path = _open_results_file("profile_vallex")
    gpu_csv = start_nvidia_smi_monitor("profile_vallex")

    header = [
        "=" * 110,
        "VALL-E-X Profile Run",
        f"Device: {device} | profile_sentence={args.profile_sentence}",
        f"Structured results: {results_path}",
        f"GPU monitor CSV: {gpu_csv or '(nvidia-smi unavailable — skipped)'}",
        "=" * 110,
    ]
    for line in header:
        _tee(results_fh, line)
    log_gpu_config(device, sink=results_fh)

    _tee(results_fh, "\nLoading VALL-E-X model...")
    t0 = time.time()
    model, codec, vocos = load_vallex_model(device)
    _tee(results_fh, f"Model loaded in {time.time() - t0:.1f}s")

    preset_path = os.path.join(_VALLEX_DIR, "presets", args.preset)
    if not os.path.exists(preset_path):
        _tee(results_fh, f"ERROR: preset {args.preset} not found at {preset_path}")
        results_fh.close()
        return

    text = SENTENCE_GROUPS[args.profile_sentence][0]
    _tee(results_fh, f"Profiling text ({args.profile_sentence}): \"{text[:80]}...\"")

    output_dir = os.path.join(_THIS_DIR, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Filter which configs to profile (default: baseline + K4/V2 — enough signal,
    # keeps total runtime under ~10 min even with profiler overhead).
    requested_configs = getattr(args, "profile_configs", None)
    if requested_configs:
        requested_names = [c.strip() for c in requested_configs.split(",") if c.strip()]
        configs_to_run = [(name, cfg) for (name, cfg) in TURBOQUANT_CONFIGS
                          if name in requested_names]
        if not configs_to_run:
            _tee(results_fh, f"ERROR: --profile-configs matched nothing. "
                             f"Requested: {requested_names}. "
                             f"Available: {[n for n, _ in TURBOQUANT_CONFIGS]}")
            results_fh.close()
            return
    else:
        configs_to_run = TURBOQUANT_CONFIGS

    # Per-config warmup so init costs don't dominate the profile
    _tee(results_fh, "\nWarmup (one per config)...")
    for config_name, tq_config in configs_to_run:
        try:
            run_generation(model, codec, vocos, "Hello.", "en", preset_path, device, tq_config)
        except Exception as e:
            _tee(results_fh, f"  warmup {config_name}: ERROR {e}")
    _tee(results_fh, "Warmup done.\n")

    for config_name, tq_config in configs_to_run:
        _tee(results_fh, f"\n{'=' * 110}")
        _tee(results_fh, f"Profiling config: {config_name}")
        _tee(results_fh, f"{'=' * 110}")
        try:
            prof, elapsed, trace_path = profile_generation(
                model, codec, vocos, text, preset_path, device,
                tq_config, output_dir, config_name, args.profile_sentence,
            )

            peak_vram_mb = read_peak_memory_mb(device)
            mem_report = None
            if tq_config is not None and hasattr(model, "_tq_cache_manager"):
                mem_report = model._tq_cache_manager.memory_report()

            _tee(results_fh, f"Elapsed: {elapsed:.2f}s  Peak VRAM: {peak_vram_mb:.0f} MB")
            if mem_report:
                _tee(results_fh,
                     f"Compressed: {mem_report['compressed_bytes']/1024**2:.2f} MB | "
                     f"Decomp prefix: {mem_report['decompressed_prefix_bytes']/1024**2:.2f} MB | "
                     f"R_theory={mem_report['theoretical_compression_ratio']:.2f}x "
                     f"R_eff={mem_report['effective_compression_ratio']:.2f}x")

            frac, label = _profile_ratio_classification(prof, elapsed)
            _tee(results_fh, label)

            # Top-20 CUDA kernels
            sort_key = "cuda_time_total" if _is_cuda(device) else "self_cpu_time_total"
            try:
                table = prof.key_averages().table(sort_by=sort_key, row_limit=20)
            except Exception as e:
                table = f"(key_averages failed: {e})"
            _tee(results_fh, f"\nTop-20 by {sort_key}:")
            _tee(results_fh, table)

            # Top-10 CPU self-time (catches CPU bottlenecks)
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

    _tee(results_fh, f"\nStructured results: {results_path}")
    results_fh.close()


def evaluate_saved_wavs(args):
    """Evaluate quality metrics on previously saved wav files (no TTS model needed)."""
    output_dir = os.path.join(_THIS_DIR, "outputs")
    if not os.path.exists(output_dir):
        print(f"ERROR: No outputs found at {output_dir}. Run generation first (without --evaluate-only).")
        return

    print("=" * 80)
    print("VALL-E-X Quality Evaluation (from saved wavs)")
    print("=" * 80)

    metrics = QualityMetrics(device=args.metrics_device)

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
                out_path = _out_path(output_dir, group_name, i, config_name)
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
    parser = argparse.ArgumentParser(description="Benchmark TurboQuant on VALL-E-X with real weights")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else
                        ("mps" if torch.backends.mps.is_available() else "cpu"),
                        help="Device to run TTS on")
    parser.add_argument("--metrics-device", default="cpu",
                        help="Device for Whisper/WavLM (default cpu; set to cuda on L40s for faster metrics)")
    parser.add_argument("--preset", default="alan.npz",
                        help="Voice preset filename under models/VALL-E-X/presets/")
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
                             "Use an empty string to profile ALL configs, which can take 60+ minutes.")
    parser.add_argument("--groups", default=",".join(SENTENCE_GROUPS.keys()),
                        help="Comma-separated sentence groups to run (default: all). "
                             "Useful on tight-VRAM GPUs: --groups short,medium to skip long sentences. "
                             "For a smoke test: --groups short.")
    parser.add_argument("--max-per-group", type=int, default=None,
                        help="Cap sentences per group (useful for smoke tests; default: run all).")
    parser.add_argument("--track-only-off", action="store_true",
                        help="Disable the Phase-1 fast path and run the legacy compression "
                             "path on-CUDA. Useful only for reconstruction-quality A/B "
                             "tests against track_only=True. Makes decode ~7x slower.")
    args = parser.parse_args()

    # Propagate track_only=False into every TurboQuantConfig if requested.
    if args.track_only_off:
        for _, cfg in TURBOQUANT_CONFIGS:
            if cfg is not None:
                cfg.track_only = False

    # Validate and apply --groups / --max-per-group filters.
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
        benchmark_vallex(args)


if __name__ == "__main__":
    main()
