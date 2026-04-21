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
        [--device cuda] [--no-quality] [--evaluate-only]
"""

import sys
import os
import time
import argparse

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
    elapsed = time.time() - start

    wav = decode_audio(codec, vocos, encoded_frames, device)

    memory_report = None
    if tq_config is not None and hasattr(model, "_tq_cache_manager"):
        memory_report = model._tq_cache_manager.memory_report()

    return wav, elapsed, memory_report


# ---------------------------------------------------------------------------
# Main benchmark loop — mirrors Qwen's per-group averaging structure
# ---------------------------------------------------------------------------

def _out_path(output_dir: str, group_name: str, idx: int, config_name: str) -> str:
    safe = config_name.replace(" ", "_").replace("/", "_")
    return os.path.join(output_dir, f"vallex_{group_name}_{idx}_{safe}.wav")


def benchmark_vallex(args):
    device = torch.device(args.device)

    print("=" * 110)
    print("VALL-E-X Real-Weights Benchmark")
    print(f"Device: {device} | Quality metrics: {not args.no_quality}")
    print("=" * 110)

    print("\nLoading VALL-E-X model...")
    t0 = time.time()
    model, codec, vocos = load_vallex_model(device)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    preset_path = os.path.join(_VALLEX_DIR, "presets", args.preset)
    if not os.path.exists(preset_path):
        print(f"ERROR: preset {args.preset} not found at {preset_path}")
        return
    print(f"Using preset: {os.path.basename(preset_path)}")

    metrics = None
    if not args.no_quality:
        print(f"\nLoading quality metrics on {args.metrics_device}...")
        metrics = QualityMetrics(device=args.metrics_device)

    print("\nWarmup generation...")
    run_generation(model, codec, vocos, "Hello.", "en", preset_path, device, None)
    print("Warmup done.\n")

    output_dir = os.path.join(_THIS_DIR, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    summary = {}

    for group_name, texts in SENTENCE_GROUPS.items():
        n = len(texts)
        print(f"\n{'=' * 110}")
        print(f"Group: {group_name} ({n} sentences)")
        print(f"{'=' * 110}")

        group_results = {name: {"rtf": [], "cer": [], "spk_sim": [], "ratio": []}
                         for name, _ in TURBOQUANT_CONFIGS}

        for i, text in enumerate(texts):
            preview = text[:50] + "..." if len(text) > 50 else text
            print(f"\n  [{i + 1}/{n}] \"{preview}\"")

            baseline_wav = None

            for config_name, tq_config in TURBOQUANT_CONFIGS:
                try:
                    wav, elapsed, mem_report = run_generation(
                        model, codec, vocos, text, "en", preset_path, device, tq_config,
                    )
                    audio_duration = len(wav) / SAMPLE_RATE
                    rtf = elapsed / audio_duration if audio_duration > 0 else float("inf")
                    group_results[config_name]["rtf"].append(rtf)

                    if mem_report:
                        group_results[config_name]["ratio"].append(mem_report["compression_ratio"])

                    status = f"RTF={rtf:.2f}"
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
                        status += f" Ratio={mem_report['compression_ratio']:.2f}x"

                    sf.write(_out_path(output_dir, group_name, i, config_name), wav, SAMPLE_RATE)
                    print(f"    {config_name:<22} {status}")

                except Exception as e:
                    print(f"    {config_name:<22} ERROR: {e}")
                    import traceback
                    traceback.print_exc()

        # Per-group averages
        print(f"\n  {'─' * 80}")
        print(f"  AVERAGES for {group_name} ({n} sentences):")
        print(f"  {'─' * 80}")
        if metrics:
            print(f"  {'Config':<22} {'Avg RTF':<10} {'Avg CER':<10} {'Avg SpkSim':<12} {'Avg Ratio':<10}")
            print(f"  {'-' * 65}")
        else:
            print(f"  {'Config':<22} {'Avg RTF':<10} {'Avg Ratio':<10}")
            print(f"  {'-' * 42}")

        for config_name, tq_config in TURBOQUANT_CONFIGS:
            r = group_results[config_name]
            avg_rtf = sum(r["rtf"]) / len(r["rtf"]) if r["rtf"] else 0
            avg_ratio = sum(r["ratio"]) / len(r["ratio"]) if r["ratio"] else (1.0 if tq_config is None else 0)
            if metrics:
                avg_cer = sum(r["cer"]) / len(r["cer"]) if r["cer"] else 0
                avg_spk = sum(r["spk_sim"]) / len(r["spk_sim"]) if r["spk_sim"] else 0
                spk_str = f"{avg_spk:.4f}" if tq_config is not None else "---"
                print(f"  {config_name:<22} {avg_rtf:<10.2f} {avg_cer:<10.2%} {spk_str:<12} {avg_ratio:<10.2f}")
            else:
                print(f"  {config_name:<22} {avg_rtf:<10.2f} {avg_ratio:<10.2f}")

        summary[group_name] = group_results

    # Final summary
    if metrics:
        print(f"\n{'=' * 110}")
        print("FINAL SUMMARY (averages across all sentences)")
        print(f"{'=' * 110}")
        print(f"{'Config':<22} ", end="")
        for group_name in SENTENCE_GROUPS:
            print(f"{'RTF':<7} {'CER':<7} {'SpkSim':<9} ", end="")
        print()
        print(f"{'':22} ", end="")
        for group_name in SENTENCE_GROUPS:
            n = len(SENTENCE_GROUPS[group_name])
            print(f"{group_name + f' ({n})':<24}", end="")
        print()

        for config_name, tq_config in TURBOQUANT_CONFIGS:
            print(f"{config_name:<22} ", end="")
            for group_name in SENTENCE_GROUPS:
                r = summary[group_name][config_name]
                avg_rtf = sum(r["rtf"]) / len(r["rtf"]) if r["rtf"] else 0
                avg_cer = sum(r["cer"]) / len(r["cer"]) if r["cer"] else 0
                avg_spk = sum(r["spk_sim"]) / len(r["spk_sim"]) if r["spk_sim"] else 0
                spk_str = f"{avg_spk:.4f}" if tq_config is not None else "---"
                print(f"{avg_rtf:<7.2f} {avg_cer:<7.1%} {spk_str:<9} ", end="")
            print()

    print(f"\nOutput audio saved to: {output_dir}/")


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
    args = parser.parse_args()

    if args.evaluate_only:
        evaluate_saved_wavs(args)
    else:
        benchmark_vallex(args)


if __name__ == "__main__":
    main()
