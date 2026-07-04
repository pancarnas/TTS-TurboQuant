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
import re
import time
import argparse
import atexit
import datetime
import shutil
import subprocess
from typing import NamedTuple, Optional

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
    N_DIM,
    NUM_HEAD,
    NUM_LAYERS,
    PREFIX_MODE,
    NUM_QUANTIZERS,
    SAMPLE_RATE,
    lang2token,
)
from data.tokenizer import AudioTokenizer
from data.collation import get_text_token_collater
from utils.g2p import PhonemeBpeTokenizer
from turboquant_cache import TurboQuantConfig

from turboquant.bench_common import (
    TRIAL_COLUMNS,
    config_bits,
    format_trial_row,
    parse_arms,
    parse_seeds,
    parse_temperatures,
    sentence_hash,
    set_global_seed,
)

# Evaluation set is shared with the Qwen benchmark for A/B parity (curated
# smoke/long + disk-backed seedtts_en / ellav_hard). See iter_eval_items.
from turboquant.eval_sentences import available_groups, iter_eval_items


# VALL-E decode-arm overrides (model.inference takes top_k/temperature directly,
# not the do_sample/subtalker kwargs the Qwen talker uses). Greedy = argmax
# (top_k=1); sampling = the model's default top_k=-100 (sample over full vocab).
def vallex_decode_params(arm: str, temperature: Optional[float] = None) -> dict:
    if arm == "greedy":
        # top_k=1 is argmax; temperature is irrelevant, so a swept value is ignored.
        return {"top_k": 1, "temperature": 1.0}
    if arm == "sampling":
        return {
            "top_k": -100,
            "temperature": 1.0 if temperature is None else temperature,
        }
    raise ValueError(f"unknown decode arm: {arm!r}")


class StageConfigs(NamedTuple):
    """TurboQuant configs per decoder stage; None disables that stage.

    The sweep entries are ``(label, StageConfigs | None)`` — None is the
    fp16 baseline. ``ar`` drives the AR KV cache (TurboQuantValleCache),
    ``nar`` the per-pass NAR K/V quantization (NARQuantizer).
    """

    ar: Optional[TurboQuantConfig] = None
    nar: Optional[TurboQuantConfig] = None

    @property
    def primary(self) -> Optional[TurboQuantConfig]:
        """The config whose bits/rw label this arm (for CSV columns)."""
        return self.ar if self.ar is not None else self.nar


def build_turboquant_configs(residual_window: int = 128) -> list:
    """The default AR-only sweep at a given residual window (Qwen twin).

    ``residual_window`` = most-recent tokens kept fp16-exact; ``rw=0`` is
    paper-faithful TurboQuant (quantize every token). Labels embed the window.
    """
    rw = residual_window
    # Value sweep at safe 4-bit keys (twin of Qwen); 3-bit keys collapse AR-codec TTS.
    return [
        ("baseline (no TQ)", None),
        (
            f"K4/V4 rw={rw}",
            StageConfigs(ar=TurboQuantConfig(key_bits=4, value_bits=4, residual_window=rw)),
        ),
        (
            f"K4/V3 rw={rw}",
            StageConfigs(ar=TurboQuantConfig(key_bits=4, value_bits=3, residual_window=rw)),
        ),
        (
            f"K4/V2 rw={rw}",
            StageConfigs(ar=TurboQuantConfig(key_bits=4, value_bits=2, residual_window=rw)),
        ),
    ]


# Default sweep (rw=128); main() rebuilds from --residual-window before consumers.
TURBOQUANT_CONFIGS = build_turboquant_configs(128)


# Same token grammar as the Qwen divergence experiment — K<kb>V<vb>@<rw> or
# fp16 — plus an optional stage prefix: 'ar:' (default), 'nar:', or 'both:'.
_CFG_RE = re.compile(r"(?:(ar|nar|both):)?[Kk](\d+)[Vv](\d+)@(\d+)$")


def parse_configs_arg(spec: str, protected_layers: int) -> list:
    """'fp16,K4V4@64,nar:K4V4@64,both:K4V4@64' -> [(label, StageConfigs|None)].

    Stage prefix selects which decoder stage(s) get quantized K/V: 'ar:'
    (default — the AR KV cache), 'nar:' (per-pass NAR quantization), or
    'both:'. Labels carry a '-nar' / '-both' suffix so wav names and CSV
    config columns stay distinct.

    Explicit-config runs are quality experiments by definition, so every TQ
    config is built with track_only=False (the lossy write-back path). The
    analytic fast path would make all arms bit-identical audio.
    """

    def _make(kb, vb, rw):
        return TurboQuantConfig(
            key_bits=kb,
            value_bits=vb,
            residual_window=rw,
            protected_layers=protected_layers,
            track_only=False,
        )

    out = []
    for tok in (t.strip() for t in spec.split(",")):
        if not tok:
            continue
        if tok.lower() in ("fp16", "baseline"):
            out.append(("fp16", None))
            continue
        m = _CFG_RE.match(tok)
        if m is None:
            raise SystemExit(
                f"bad config token {tok!r}; expected e.g. K4V4@64, "
                "nar:K4V4@64, both:K4V4@64, or fp16"
            )
        stage = m.group(1) or "ar"
        kb, vb, rw = int(m.group(2)), int(m.group(3)), int(m.group(4))
        # Separate config instances per stage — never share a mutable config.
        entry = StageConfigs(
            ar=_make(kb, vb, rw) if stage in ("ar", "both") else None,
            nar=_make(kb, vb, rw) if stage in ("nar", "both") else None,
        )
        suffix = "" if stage == "ar" else f"-{stage}"
        out.append((f"K{kb}V{vb}@{rw}{suffix}", entry))
    if not out:
        raise SystemExit("--configs given but no valid tokens parsed")
    return out


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

    @staticmethod
    def _normalize_for_wer(text: str) -> str:
        """Lowercase, strip punctuation (keeping in-word apostrophes), collapse
        whitespace — the standard TTS-eval WER protocol, so casing and
        punctuation differences don't count as word errors."""
        text = re.sub(r"[^\w\s']", " ", text.lower())
        return " ".join(text.split())

    def whisper_scores(
        self, wav: np.ndarray, sr: int, reference_text: str
    ) -> tuple[float, float, str]:
        """One Whisper pass -> (CER, WER, transcript).

        CER is computed on the raw strings (backward compatible with all prior
        runs); WER on normalized text (see _normalize_for_wer).
        """
        self._load_whisper()
        from jiwer import cer, wer

        wav = wav.astype(np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        result = self._whisper.transcribe(wav)
        transcript = result["text"].strip()
        ref = reference_text.strip()
        if not ref:
            return 0.0, 0.0, transcript
        ref_norm = self._normalize_for_wer(ref)
        hyp_norm = self._normalize_for_wer(transcript)
        word_error = float(wer(ref_norm, hyp_norm)) if ref_norm else 0.0
        return float(cer(ref, transcript)), word_error, transcript

    # Sentinel values for self._wavlm_model:
    #   None  — not loaded yet
    #   False — load failed (e.g., tensorflow/CUDA ABI conflict); skip speaker-sim
    #   <obj> — loaded successfully

    def _load_wavlm(self) -> bool:
        """Load WavLM lazily. Return True on success, False on failure.

        Load failures typically come from transformers transitively importing
        tensorflow on systems where TF's bundled cuDNN conflicts with the
        system one. If that happens we degrade gracefully: CER still works,
        speaker-sim is skipped with a one-shot warning.
        """
        if self._wavlm_model is False:
            return False
        if self._wavlm_model is None:
            try:
                from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

                self._wavlm_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                    "microsoft/wavlm-base-plus-sv"
                )
                self._wavlm_model = (
                    WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv")
                    .to(self._device)
                    .eval()
                )
                # Device for WavLM; flips to CPU on a CUDA OOM (long audio under
                # multi-shard contention) so a trial row is never dropped.
                self._wavlm_device = self._device
            except Exception as e:
                print(
                    f"WARNING: WavLM failed to load ({type(e).__name__}: {e}). "
                    f"Speaker similarity will be skipped. If this is a "
                    f"tensorflow/cuDNN conflict, try: pip uninstall -y tensorflow tensorflow-cpu"
                )
                self._wavlm_model = False
                return False
        return True

    def _wavlm_embed(self, inputs):
        inputs = {k: v.to(self._wavlm_device) for k, v in inputs.items()}
        with torch.no_grad():
            emb = self._wavlm_model(**inputs).embeddings
            emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb.squeeze().cpu().numpy()

    def speaker_embedding(self, wav: np.ndarray, sr: int):
        if not self._load_wavlm():
            return None
        wav = wav.astype(np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
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
    ):
        """Return cosine similarity, or None if WavLM is unavailable."""
        emb_a = self.speaker_embedding(wav_a, sr_a)
        if emb_a is None:
            return None
        emb_b = self.speaker_embedding(wav_b, sr_b)
        if emb_b is None:
            return None
        return float(np.dot(emb_a, emb_b))


# ---------------------------------------------------------------------------
# Model loading & generation
# ---------------------------------------------------------------------------


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
    return fh, path


def _open_csv_file(prefix: str, columns: list) -> tuple:
    """Open a real ``.csv`` (header row, no '#') for downstream pandas analysis."""
    repo_root = os.path.dirname(os.path.dirname(_VALLEX_DIR))
    results_dir = os.path.join(repo_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(results_dir, f"{prefix}_{ts}.csv")
    fh = open(path, "a", buffering=1, encoding="utf-8")
    fh.write(",".join(columns) + "\n")
    return fh, path


def _write_result_line(fh, **kw):
    """Append one per-trial CSV row using the shared TRIAL_COLUMNS schema."""
    fh.write(format_trial_row(kw) + "\n")


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

    model = (
        VALLE(
            N_DIM,
            NUM_HEAD,
            NUM_LAYERS,
            norm_first=True,
            add_prenet=False,
            prefix_mode=PREFIX_MODE,
            share_embedding=True,
            nar_scale_factor=1.0,
            prepend_bos=True,
            num_quantizers=NUM_QUANTIZERS,
        )
        .half()
        .to(device)
    )

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
def run_generation(
    model,
    codec,
    vocos,
    text,
    language,
    preset_path,
    device,
    tq_config,
    seed=None,
    decode_params=None,
    deterministic=False,
):
    """Run one generation end-to-end.

    Returns (wav, elapsed, memory_report, peak_vram_mb, n_ar_tokens).
    Wall time covers GPU completion via an explicit sync before/after inference.

    ``seed`` (if given) is applied via set_global_seed right before inference so
    baseline and every compressed config share the same random draw for this
    (sentence, seed) — the paired-comparison control. ``decode_params`` overrides
    top_k/temperature (greedy vs sampling arm).
    """
    dp = {"top_k": -100, "temperature": 1.0}
    if decode_params:
        dp.update(decode_params)
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

    if seed is not None:
        set_global_seed(seed, deterministic=deterministic)

    start = time.time()
    encoded_frames = model.inference(
        text_tokens.to(device),
        text_tokens_lens.to(device),
        audio_prompts,
        enroll_x_lens=enroll_x_lens,
        top_k=dp["top_k"],
        temperature=dp["temperature"],
        prompt_language=lang_pr,
        text_language=langs,
        turboquant_config=tq_config.ar if tq_config is not None else None,
        turboquant_config_nar=tq_config.nar if tq_config is not None else None,
    )
    if _is_cuda(device):
        torch.cuda.synchronize(device)
    elapsed = time.time() - start

    peak_vram_mb = read_peak_memory_mb(device)
    # encoded_frames shape (B=1, T, num_quantizers) — T is the AR-decoded token count.
    n_ar_tokens = int(encoded_frames.shape[1]) if encoded_frames.ndim >= 2 else 0

    # AR/NAR phase split (ms) — populated by vallex.py when CUDA timing is available.
    ar_elapsed_s = getattr(model, "_ar_elapsed_ms", 0.0) / 1000.0
    nar_elapsed_s = getattr(model, "_nar_elapsed_ms", 0.0) / 1000.0

    wav = decode_audio(codec, vocos, encoded_frames, device)

    # Memory report exists only for the AR cache manager; NAR-only arms have
    # nothing cached, so their memory fields stay empty by design.
    memory_report = None
    if (
        tq_config is not None
        and tq_config.ar is not None
        and hasattr(model, "_tq_cache_manager")
        and model._tq_cache_manager is not None
    ):
        memory_report = model._tq_cache_manager.memory_report()

    return (
        wav,
        elapsed,
        memory_report,
        peak_vram_mb,
        n_ar_tokens,
        ar_elapsed_s,
        nar_elapsed_s,
    )


# ---------------------------------------------------------------------------
# Main benchmark loop — mirrors Qwen's per-group averaging structure
# ---------------------------------------------------------------------------


def _out_path(
    output_dir: str,
    group_name: str,
    idx: int,
    config_name: str,
    arm: str = "sampling",
    seed: int = 0,
    temperature=None,
) -> str:
    """Path a generated wav was saved to (must mirror the save in the sweep)."""
    safe = config_name.replace(" ", "_").replace("/", "_")
    tsuf = "" if temperature is None else f"_t{temperature}"
    return os.path.join(
        output_dir, f"vallex_{group_name}_{idx}_{arm}_s{seed}{tsuf}_{safe}.wav"
    )


def _tee(fh, msg):
    """Print to stdout and to the results file (if any)."""
    print(msg)
    if fh is not None:
        fh.write(msg + "\n")


def _load_reference_clip(item, metrics):
    """Load the reference clip for ground-truth SpkSim (ground truth, else prompt).

    NOTE: VALL-E generates with the preset voice (it does not clone an arbitrary
    reference WAV), so spk_sim_ref here mostly measures preset-vs-reference voice
    mismatch — informative, but not a cloning-fidelity number. Returns (wav, sr)
    or (None, None).
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


def _vallex_empty_group_results() -> dict:
    metric_keys = (
        "rtf",
        "ar_rtf",
        "nar_rtf",
        "cer",
        "wer",
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


def _vallex_mem_metrics(mem_report) -> dict:
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
        "sim_compressed_mb": mem_report.get(
            "simulated_compressed_bytes", mem_report.get("compressed_bytes", 0)
        )
        / (1024**2),
        "realized_mb": mem_report.get(
            "realized_fp16_bytes", mem_report.get("total_bytes", 0)
        )
        / (1024**2),
    }


def _vallex_sweep_sentence_seed(
    model,
    codec,
    vocos,
    metrics,
    preset_path,
    item,
    group_name,
    idx,
    shash,
    arm,
    seed,
    temperature,
    decode_params,
    group_results,
    output_dir,
    results_fh,
    trial_fh,
    device,
    deterministic,
    save_wav,
):
    """Run all configs for one (sentence, seed, temperature), paired on the same draw."""
    text = item.text
    baseline_wav = None
    ref_wav, ref_sr = _load_reference_clip(item, metrics)
    for config_name, tq_config in TURBOQUANT_CONFIGS:
        key_bits, value_bits, residual_window = config_bits(
            tq_config.primary if tq_config is not None else None
        )
        try:
            (
                wav,
                elapsed,
                mem_report,
                peak_vram_mb,
                n_ar_tokens,
                ar_elapsed_s,
                nar_elapsed_s,
            ) = run_generation(
                model,
                codec,
                vocos,
                text,
                "en",
                preset_path,
                device,
                tq_config,
                seed=seed,
                decode_params=decode_params,
                deterministic=deterministic,
            )
            audio_duration = len(wav) / SAMPLE_RATE
            rtf = elapsed / audio_duration if audio_duration > 0 else float("inf")
            ar_rtf = (
                ar_elapsed_s / audio_duration
                if (audio_duration > 0 and ar_elapsed_s > 0)
                else 0.0
            )
            nar_rtf = (
                nar_elapsed_s / audio_duration
                if (audio_duration > 0 and nar_elapsed_s > 0)
                else 0.0
            )
            tokens_per_sec = n_ar_tokens / elapsed if elapsed > 0 else 0.0
            mm = _vallex_mem_metrics(mem_report)

            r = group_results[config_name]
            r["rtf"].append(rtf)
            r["ar_rtf"].append(ar_rtf)
            r["nar_rtf"].append(nar_rtf)
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
            word_error_rate = None
            spk_sim = None
            spk_sim_ref = None
            transcript_len = None
            if metrics:
                error_rate, word_error_rate, transcript = metrics.whisper_scores(
                    wav, SAMPLE_RATE, text
                )
                transcript_len = len(transcript)
                r["cer"].append(error_rate)
                r["wer"].append(word_error_rate)
                if tq_config is None:
                    baseline_wav = wav
                elif baseline_wav is not None:
                    spk_sim = metrics.speaker_cosine_similarity(
                        baseline_wav, SAMPLE_RATE, wav, SAMPLE_RATE
                    )
                    if spk_sim is not None:
                        r["spk_sim"].append(spk_sim)
                if ref_wav is not None:
                    spk_sim_ref = metrics.speaker_cosine_similarity(
                        ref_wav, ref_sr, wav, SAMPLE_RATE
                    )
                    if spk_sim_ref is not None:
                        r["spk_sim_ref"].append(spk_sim_ref)

            if save_wav:
                safe = config_name.replace(" ", "_").replace("/", "_")
                tsuf = "" if temperature is None else f"_t{temperature}"
                sf.write(
                    os.path.join(
                        output_dir,
                        f"vallex_{group_name}_{idx}_{arm}_s{seed}{tsuf}_{safe}.wav",
                    ),
                    wav,
                    SAMPLE_RATE,
                )

            status = (
                f"RTF={rtf:.2f} VRAM={peak_vram_mb:.0f}MB tok/s={tokens_per_sec:.1f}"
            )
            if error_rate is not None:
                status += f" CER={error_rate:.1%}"
                if word_error_rate is not None:
                    status += f" WER={word_error_rate:.1%}"
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


def _vallex_print_group_averages(
    results_fh, metrics, group_name, group_results
) -> None:
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


def _vallex_sweep_arm(
    model,
    codec,
    vocos,
    metrics,
    preset_path,
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
) -> dict:
    """Full group×sentence×seed×temperature×config sweep for one arm. Returns summary.

    Temperature only varies for the sampling arm (greedy ignores it → single None anchor).
    """
    temps = temperatures if (arm == "sampling" and temperatures) else [None]
    _tee(results_fh, f"\n{'#' * 110}")
    temp_note = f" temps={temps}" if temps != [None] else ""
    _tee(results_fh, f"DECODE ARM: {arm}{temp_note}")
    _tee(results_fh, f"{'#' * 110}")
    summary = {}
    # Round-robin shard counter over (group, sentence, seed, temperature) cells;
    # the config loop stays whole inside each cell (baseline-as-ref pairing).
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
        group_results = _vallex_empty_group_results()
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
                    decode_params = vallex_decode_params(arm, temperature=temp)
                    _vallex_sweep_sentence_seed(
                        model,
                        codec,
                        vocos,
                        metrics,
                        preset_path,
                        item,
                        group_name,
                        i,
                        shash,
                        arm,
                        seed,
                        temp,
                        decode_params,
                        group_results,
                        output_dir,
                        results_fh,
                        trial_fh,
                        device,
                        deterministic,
                        save_wav=(seed == seeds[0] and temp == temps[0]),
                    )
        _vallex_print_group_averages(results_fh, metrics, group_name, group_results)
        summary[group_name] = group_results
    return summary


def _vallex_print_arm_summaries(
    results_fh, metrics, arm, active_groups, summary
) -> None:
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


def benchmark_vallex(args):
    device = torch.device(args.device)
    arms = parse_arms(args.decode)
    seeds = args.seeds

    num_shards = getattr(args, "num_shards", 1)
    shard_id = getattr(args, "shard_id", 0)
    data_dir = getattr(args, "data_dir", None)
    run_tag = getattr(args, "run_tag", "") or ""
    trial_tag = "vallex_trials" if num_shards == 1 else f"vallex_trials_shard{shard_id}"
    if run_tag:
        trial_tag = f"{trial_tag}_{run_tag}"

    results_fh, results_path = _open_results_file("benchmark_vallex")
    trial_fh, trial_path = _open_csv_file(trial_tag, TRIAL_COLUMNS)
    gpu_csv = start_nvidia_smi_monitor("benchmark_vallex")

    header = [
        "=" * 110,
        "VALL-E-X Rigorous Benchmark (paired-seed, multi-arm)",
        f"Device: {device} | Quality metrics: {not args.no_quality}",
        f"Decode arms: {arms} | Seeds: {seeds} | Deterministic: {args.deterministic}",
        f"Temperature sweep (sampling arm): {args.temperatures or 'off (model default)'}",
        "Pairing: each (sentence, seed) reseeds RNG identically before every config.",
        f"Human log: {results_path}",
        f"Per-trial CSV: {trial_path}",
        f"GPU monitor CSV: {gpu_csv or '(nvidia-smi unavailable — skipped)'}",
        "=" * 110,
    ]
    for line in header:
        _tee(results_fh, line)

    log_gpu_config(device, sink=results_fh)

    _tee(results_fh, "\nLoading VALL-E-X model...")
    t0 = time.time()
    if _is_cuda(device):
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    model, codec, vocos = load_vallex_model(device)
    if _is_cuda(device):
        torch.cuda.synchronize(device)
    _tee(results_fh, f"Model loaded in {time.time() - t0:.1f}s")
    model_weight_mb = read_peak_memory_mb(device)
    _tee(results_fh, f"Model weights + load-time VRAM: {model_weight_mb:.0f} MB")

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
            run_generation(
                model, codec, vocos, "Hello.", "en", preset_path, device, tq_config
            )
        except Exception as e:
            _tee(results_fh, f"  warmup {config_name}: ERROR {e}")
    _tee(results_fh, "Warmup done.\n")

    output_dir = os.path.join(_THIS_DIR, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    active_groups = getattr(args, "active_groups", available_groups())
    max_per_group = getattr(args, "max_per_group", None)

    for arm in arms:
        summary = _vallex_sweep_arm(
            model,
            codec,
            vocos,
            metrics,
            preset_path,
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
        )
        _vallex_print_arm_summaries(results_fh, metrics, arm, active_groups, summary)

    trial_fh.close()
    _tee(results_fh, f"\nOutput audio saved to: {output_dir}/")
    _tee(results_fh, f"Per-trial CSV (source of truth): {trial_path}")
    results_fh.close()


# ---------------------------------------------------------------------------
# Profiling mode
# ---------------------------------------------------------------------------


@torch.no_grad()
def profile_generation(
    model,
    codec,
    vocos,
    text,
    preset_path,
    device,
    tq_config,
    output_dir,
    config_name,
    group_name,
):
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
    with profile(
        activities=activities,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        model.inference(
            text_tokens.to(device),
            text_tokens_lens.to(device),
            audio_prompts,
            enroll_x_lens=enroll_x_lens,
            top_k=-100,
            temperature=1.0,
            prompt_language=lang_pr,
            text_language=langs,
            turboquant_config=tq_config.ar if tq_config is not None else None,
            turboquant_config_nar=tq_config.nar if tq_config is not None else None,
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
        label = (
            f"LAUNCH-BOUND (cuda_time/wall = {frac:.1%}): batching would likely help. "
            f"Follow-up: patch vallex.py:492 for batch>1."
        )
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
    if _is_cuda(device):
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    model, codec, vocos = load_vallex_model(device)
    if _is_cuda(device):
        torch.cuda.synchronize(device)
    _tee(results_fh, f"Model loaded in {time.time() - t0:.1f}s")
    model_weight_mb = read_peak_memory_mb(device)
    _tee(results_fh, f"Model weights + load-time VRAM: {model_weight_mb:.0f} MB")

    preset_path = os.path.join(_VALLEX_DIR, "presets", args.preset)
    if not os.path.exists(preset_path):
        _tee(results_fh, f"ERROR: preset {args.preset} not found at {preset_path}")
        results_fh.close()
        return

    text = iter_eval_items(
        [args.profile_sentence],
        max_per_group=1,
        data_dir=getattr(args, "data_dir", None),
    )[0].text
    _tee(results_fh, f'Profiling text ({args.profile_sentence}): "{text[:80]}..."')

    output_dir = os.path.join(_THIS_DIR, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Filter which configs to profile (default: baseline + K4/V2 — enough signal,
    # keeps total runtime under ~10 min even with profiler overhead).
    requested_configs = getattr(args, "profile_configs", None)
    if requested_configs:
        requested_names = [c.strip() for c in requested_configs.split(",") if c.strip()]
        configs_to_run = [
            (name, cfg) for (name, cfg) in TURBOQUANT_CONFIGS if name in requested_names
        ]
        if not configs_to_run:
            _tee(
                results_fh,
                f"ERROR: --profile-configs matched nothing. "
                f"Requested: {requested_names}. "
                f"Available: {[n for n, _ in TURBOQUANT_CONFIGS]}",
            )
            results_fh.close()
            return
    else:
        configs_to_run = TURBOQUANT_CONFIGS

    # Per-config warmup so init costs don't dominate the profile
    _tee(results_fh, "\nWarmup (one per config)...")
    for config_name, tq_config in configs_to_run:
        try:
            run_generation(
                model, codec, vocos, "Hello.", "en", preset_path, device, tq_config
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
                codec,
                vocos,
                text,
                preset_path,
                device,
                tq_config,
                output_dir,
                config_name,
                args.profile_sentence,
            )

            peak_vram_mb = read_peak_memory_mb(device)
            mem_report = None
            if (
                tq_config is not None
                and tq_config.ar is not None
                and hasattr(model, "_tq_cache_manager")
                and model._tq_cache_manager is not None
            ):
                mem_report = model._tq_cache_manager.memory_report()

            _tee(
                results_fh, f"Elapsed: {elapsed:.2f}s  Peak VRAM: {peak_vram_mb:.0f} MB"
            )
            if mem_report:
                _tee(
                    results_fh,
                    f"Compressed: {mem_report['compressed_bytes'] / 1024**2:.2f} MB | "
                    f"Decomp prefix: {mem_report['decompressed_prefix_bytes'] / 1024**2:.2f} MB | "
                    f"R_theory={mem_report['theoretical_compression_ratio']:.2f}x "
                    f"R_eff={mem_report['effective_compression_ratio']:.2f}x",
                )

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
            # Release the profiler (retains ~1KB per event) and the TurboQuant
            # cache before moving to the next config. Prevents host-RAM OOM on
            # long profiles (Qwen at 1.7B has ~700k kernel events per run).
            import gc

            try:
                del prof
            except NameError:
                pass
            if hasattr(model, "_tq_cache_manager"):
                model._tq_cache_manager = None
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
    print("VALL-E-X Quality Evaluation (from saved wavs)")
    print("=" * 80)

    metrics = QualityMetrics(device=args.metrics_device)

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
                # Note: assumes the default sweep shape (sampling arm, seed 0,
                # no temperature suffix). For full post-hoc scoring incl. WER
                # use tools/score_wav_dir.py instead.
                out_path = _out_path(output_dir, group_name, i, config_name)
                if not os.path.exists(out_path):
                    continue
                wav, sr = sf.read(out_path)
                wav = wav.astype(np.float32)
                error_rate, _, _ = metrics.whisper_scores(wav, sr, text)
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

        for config_name, tq_config in TURBOQUANT_CONFIGS:
            r = group_results[config_name]
            avg_cer = sum(r["cer"]) / len(r["cer"]) if r["cer"] else 0
            avg_spk = sum(r["spk_sim"]) / len(r["spk_sim"]) if r["spk_sim"] else 0
            spk_str = f"{avg_spk:.4f}" if tq_config is not None else "---"
            print(f"{config_name:<22} {avg_cer:<10.2%} {spk_str:<12}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark TurboQuant on VALL-E-X with real weights"
    )
    parser.add_argument(
        "--device",
        default="cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu"),
        help="Device to run TTS on",
    )
    parser.add_argument(
        "--metrics-device",
        default="cpu",
        help="Device for Whisper/WavLM (default cpu; set to cuda on L40s for faster metrics)",
    )
    parser.add_argument(
        "--preset",
        default="alan.npz",
        help="Voice preset filename under models/VALL-E-X/presets/",
    )
    parser.add_argument(
        "--no-quality",
        action="store_true",
        help="Skip quality metrics (Whisper CER, WavLM similarity)",
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
        "Use an empty string to profile ALL configs, which can take 60+ minutes.",
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
        help="Directory with the standard eval sets (en/*.lst + prompt-wavs + "
        "ellav_hard.txt) from tools/fetch_eval_data.py. Needed for seedtts_en / "
        "ellav_hard. NOTE: VALL-E uses the preset voice (no arbitrary-WAV cloning), "
        "so ground-truth spk_sim_ref mainly reflects preset-vs-reference mismatch.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split the work-list across N parallel workers (round-robin over "
        "cells). Each shard writes its own trials CSV; use WORKERS=1 for perf.",
    )
    parser.add_argument(
        "--shard-id", type=int, default=0, help="This worker's shard index in [0, N)."
    )
    parser.add_argument(
        "--run-tag",
        default="",
        help="Shared label embedded in the trials CSV name so a parallel launch's "
        "shards group into one run for analyze --trials-glob.",
    )
    parser.add_argument(
        "--residual-window",
        type=int,
        default=128,
        help="Most-recent tokens kept fp16-exact; older tokens are quantized. "
        "rw=0 is paper-faithful TurboQuant (quantize every token). Default 128. "
        "Ignored when --configs is given (each token carries its own rw).",
    )
    parser.add_argument(
        "--configs",
        default="",
        help="Explicit per-config sweep, e.g. "
        "'fp16,K4V4@64,nar:K4V4@64,both:K4V4@64' "
        "(K<key_bits>V<value_bits>@<residual_window>; 'fp16' = no-TQ baseline; "
        "optional stage prefix 'ar:' (default) / 'nar:' / 'both:' selects which "
        "decoder stage gets quantized K/V). Overrides the default sweep and "
        "--residual-window, and FORCES track_only=False (lossy write-back "
        "path) on every TQ config so quality differences are real.",
    )
    parser.add_argument(
        "--protected-layers",
        type=int,
        default=2,
        help="First N and last N decoder layers keep 8-bit K/V (TurboQuantConfig "
        "default 2 -> layers 0,1,10,11 of VALL-E-X's 12). Applied to every TQ "
        "config in the sweep. Use 0 for a no-protection run.",
    )
    parser.add_argument(
        "--track-only-off",
        action="store_true",
        help="Disable the Phase-1 fast path and run the legacy compression "
        "path on-CUDA. Useful only for reconstruction-quality A/B "
        "tests against track_only=True. Makes decode ~7x slower.",
    )
    parser.add_argument(
        "--seeds",
        default="0,1,2,3,4",
        help="Comma-separated generation seeds for repetition (default: 5 seeds). "
        "Each (sentence, seed) reseeds every config identically (paired design).",
    )
    parser.add_argument(
        "--decode",
        default="both",
        choices=["sampling", "greedy", "both"],
        help="Decode arm(s): 'sampling' (top_k=-100, stochastic), 'greedy' "
        "(top_k=1, deterministic argmax), or 'both' (default).",
    )
    parser.add_argument(
        "--temperatures",
        default="",
        help="Comma-separated sampling temperatures to sweep, e.g. '0.7,1.0,1.2'. "
        "Applies to the SAMPLING arm only (greedy ignores temperature). Empty "
        "(default) = no sweep. Only meaningful together with --track-only-off.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Request deterministic CUDA kernels (slower; run in a fresh process).",
    )
    args = parser.parse_args()

    # Rebuild the sweep before any consumer runs: either the explicit
    # --configs list (per-config rw, lossy path forced) or the legacy default
    # sweep at the shared --residual-window.
    if args.residual_window < 0:
        parser.error(f"--residual-window must be >= 0, got {args.residual_window}")
    if args.protected_layers < 0:
        parser.error(f"--protected-layers must be >= 0, got {args.protected_layers}")
    global TURBOQUANT_CONFIGS
    if args.configs:
        TURBOQUANT_CONFIGS = parse_configs_arg(args.configs, args.protected_layers)
        print(
            f"[configs] explicit sweep: {[n for n, _ in TURBOQUANT_CONFIGS]} "
            f"(protected_layers={args.protected_layers}, track_only=False forced)"
        )
    else:
        TURBOQUANT_CONFIGS = build_turboquant_configs(args.residual_window)
        for _, entry in TURBOQUANT_CONFIGS:
            if entry is not None:
                for cfg in (entry.ar, entry.nar):
                    if cfg is not None:
                        cfg.protected_layers = args.protected_layers

    # Propagate track_only=False into every TurboQuantConfig if requested.
    if args.track_only_off:
        for _, entry in TURBOQUANT_CONFIGS:
            if entry is not None:
                for cfg in (entry.ar, entry.nar):
                    if cfg is not None:
                        cfg.track_only = False

    # Validate and apply --groups / --max-per-group filters.
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
        benchmark_vallex(args)


if __name__ == "__main__":
    main()
