"""Benchmark TurboQuant KV cache on Qwen3-TTS with real model weights.

Loads the actual Qwen3-TTS model, generates speech with and without
TurboQuant compression, and compares:
  - Latency and memory usage
  - Whisper CER (character error rate)
  - WavLM speaker cosine similarity vs baseline

Usage:
    python models/Qwen3-TTS/benchmarks/benchmark_qwen3tts_real.py \
        [--model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice] \
        [--device mps] \
        [--dtype bfloat16] \
        [--no-quality]  # skip quality metrics (faster)

Requires:
    pip install openai-whisper jiwer
"""

import os
import time
import argparse

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


def run_generation(model, text, language, speaker, tq_config):
    """Run a single generation, return (wavs, sr, elapsed, memory_report)."""
    kwargs = {}
    if tq_config is not None:
        kwargs["turboquant_config"] = tq_config

    start = time.time()
    wavs, sr = model.generate_custom_voice(
        text=text,
        language=language,
        speaker=speaker,
        **kwargs,
    )
    elapsed = time.time() - start

    memory_report = None
    if tq_config is not None and hasattr(model.model, "last_kv_cache"):
        cache = model.model.last_kv_cache
        if hasattr(cache, "memory_report"):
            memory_report = cache.memory_report()

    return wavs, sr, elapsed, memory_report


def benchmark_qwen3tts(args):
    print("=" * 110)
    print("Qwen3-TTS Real-Weights Benchmark")
    print(f"Model: {args.model}")
    print(f"Device: {args.device} | Dtype: {args.dtype} | Quality metrics: {not args.no_quality}")
    print("=" * 110)

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(args.dtype, torch.bfloat16)

    # Load TTS model
    print("\nLoading Qwen3-TTS model...")
    t0 = time.time()
    model = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map=args.device,
        dtype=dtype,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")

    speakers = model.get_supported_speakers()
    speaker = speakers[0] if speakers else "Ryan"
    print(f"Using speaker: {speaker}")

    # Load quality metrics
    metrics = None
    if not args.no_quality:
        # Use CPU for metrics models to avoid MPS memory pressure
        metrics_device = "cpu"
        print(f"\nLoading quality metrics on {metrics_device}...")
        metrics = QualityMetrics(device=metrics_device)

    # Warmup
    print("\nWarmup generation...")
    run_generation(model, "Hello world.", "English", speaker, None)
    print("Warmup done.\n")

    output_dir = os.path.join(_THIS_DIR, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Collect results per group for summary
    summary = {}

    for group_name, texts in SENTENCE_GROUPS.items():
        n = len(texts)
        print(f"\n{'=' * 110}")
        print(f"Group: {group_name} ({n} sentences)")
        print(f"{'=' * 110}")

        # Per-config accumulators
        group_results = {name: {"rtf": [], "cer": [], "spk_sim": []} for name, _ in TURBOQUANT_CONFIGS}

        for i, text in enumerate(texts):
            text_preview = text[:50] + "..." if len(text) > 50 else text
            print(f"\n  [{i+1}/{n}] \"{text_preview}\"")

            baseline_wav = None
            baseline_sr = None

            for config_name, tq_config in TURBOQUANT_CONFIGS:
                try:
                    wavs, sr, elapsed, mem_report = run_generation(
                        model, text, "English", speaker, tq_config,
                    )
                    wav = wavs[0]
                    audio_duration = len(wav) / sr
                    rtf = elapsed / audio_duration if audio_duration > 0 else float("inf")
                    group_results[config_name]["rtf"].append(rtf)

                    if metrics:
                        error_rate, _ = metrics.whisper_cer(wav, sr, text)
                        group_results[config_name]["cer"].append(error_rate)

                        if tq_config is None:
                            baseline_wav = wav
                            baseline_sr = sr
                        else:
                            spk_sim = metrics.speaker_cosine_similarity(
                                baseline_wav, baseline_sr, wav, sr
                            )
                            group_results[config_name]["spk_sim"].append(spk_sim)

                    # Save audio
                    out_path = os.path.join(
                        output_dir,
                        f"qwen_{group_name}_{i}_{config_name.replace(' ', '_').replace('/', '_')}.wav",
                    )
                    sf.write(out_path, wav, sr)

                    status = f"RTF={rtf:.2f}"
                    if metrics and tq_config is not None:
                        status += f" CER={error_rate:.1%} SpkSim={spk_sim:.4f}"
                    elif metrics:
                        status += f" CER={error_rate:.1%}"
                    print(f"    {config_name:<22} {status}")

                except Exception as e:
                    print(f"    {config_name:<22} ERROR: {e}")

        # Print group averages
        print(f"\n  {'─' * 80}")
        print(f"  AVERAGES for {group_name} ({n} sentences):")
        print(f"  {'─' * 80}")
        if metrics:
            print(f"  {'Config':<22} {'Avg RTF':<10} {'Avg CER':<10} {'Avg SpkSim':<12}")
            print(f"  {'-' * 55}")
        else:
            print(f"  {'Config':<22} {'Avg RTF':<10}")
            print(f"  {'-' * 32}")

        for config_name, tq_config in TURBOQUANT_CONFIGS:
            r = group_results[config_name]
            avg_rtf = sum(r["rtf"]) / len(r["rtf"]) if r["rtf"] else 0
            if metrics:
                avg_cer = sum(r["cer"]) / len(r["cer"]) if r["cer"] else 0
                avg_spk = sum(r["spk_sim"]) / len(r["spk_sim"]) if r["spk_sim"] else 0
                spk_str = f"{avg_spk:.4f}" if tq_config is not None else "---"
                print(f"  {config_name:<22} {avg_rtf:<10.2f} {avg_cer:<10.2%} {spk_str:<12}")
            else:
                print(f"  {config_name:<22} {avg_rtf:<10.2f}")

        summary[group_name] = group_results

    # Final summary table
    if metrics:
        print(f"\n{'=' * 110}")
        print("FINAL SUMMARY (averages across all sentences)")
        print(f"{'=' * 110}")
        print(f"{'Config':<22} ", end="")
        for group_name in SENTENCE_GROUPS:
            n = len(SENTENCE_GROUPS[group_name])
            print(f"{'RTF':<7} {'CER':<7} {'SpkSim':<9} ", end="")
        print()
        print(f"{'':22} ", end="")
        for group_name in SENTENCE_GROUPS:
            n = len(SENTENCE_GROUPS[group_name])
            print(f"{'─' * 23} ", end="")
        print()
        print(f"{'':22} ", end="")
        for group_name in SENTENCE_GROUPS:
            n = len(SENTENCE_GROUPS[group_name])
            print(f"{group_name+f' ({n})':<24}", end="")
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

    # Attention similarity test — all sentences per group, averaged
    print(f"\n{'=' * 110}")
    print("KV Cache Reconstruction Quality (attention similarity)")
    print(f"{'=' * 110}")

    configs = [("K4/V2", 4, 2), ("K3/V3", 3, 3), ("K3/V2", 3, 2), ("K2/V2", 2, 2)]
    all_group_avgs = {}

    for group_name, texts in SENTENCE_GROUPS.items():
        n = len(texts)
        print(f"\n  [{group_name}] ({n} sentences)")
        group_sims = {name: [] for name, _, _ in configs}

        for i, text in enumerate(texts):
            result = measure_attention_similarity(model, text, "English", speaker)
            for name in result:
                group_sims[name].append(result[name])

        print(f"\n  {'Config':<12} {'Avg Similarity':<16} {'Min':<10} {'Max':<10}")
        print(f"  {'-' * 48}")
        for name, _, _ in configs:
            vals = group_sims[name]
            if vals:
                avg = sum(vals) / len(vals)
                print(f"  {name:<12} {avg:<16.4f} {min(vals):<10.4f} {max(vals):<10.4f}")
                all_group_avgs.setdefault(name, {})[group_name] = avg

    # Final summary
    print(f"\n  {'─' * 60}")
    print(f"  Attention Similarity Summary:")
    print(f"  {'Config':<12}", end="")
    for group_name in SENTENCE_GROUPS:
        print(f" {group_name:<12}", end="")
    print()
    for name, _, _ in configs:
        print(f"  {name:<12}", end="")
        for group_name in SENTENCE_GROUPS:
            val = all_group_avgs.get(name, {}).get(group_name, 0)
            print(f" {val:<12.4f}", end="")
        print()


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
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else
                        ("cuda" if torch.cuda.is_available() else "cpu"),
                        help="Device to run on")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"],
                        help="Model dtype")
    parser.add_argument("--no-quality", action="store_true",
                        help="Skip quality metrics (Whisper CER, WavLM similarity)")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Skip generation, evaluate saved wavs from a previous run")
    args = parser.parse_args()

    if args.evaluate_only:
        evaluate_saved_wavs(args)
    else:
        benchmark_qwen3tts(args)


if __name__ == "__main__":
    main()
