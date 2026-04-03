"""Qwen3-TTS generation with TurboQuant KV cache compression.

Generates speech with and without TurboQuant, prints memory stats,
and saves output audio for comparison.

Usage:
    python examples/qwen3_tts_turboquant.py
"""

import torch
import soundfile as sf

from qwen_tts import Qwen3TTSModel
from turboquant import TurboQuantConfig


def main():
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Loading Qwen3-TTS on {device}...")
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device_map=device,
        dtype=torch.bfloat16,
    )

    text = "There was a famine in the land, so Abram went down to Egypt to stay there for a while."
    language = "English"
    speaker = model.get_supported_speakers()[0]

    # --- Without TurboQuant (baseline) ---
    print(f"\nGenerating with baseline (no compression)...")
    wavs, sr = model.generate_custom_voice(
        text=text,
        language=language,
        speaker=speaker,
    )
    sf.write("output_baseline.wav", wavs[0], sr)
    print(f"  Saved output_baseline.wav ({len(wavs[0]) / sr:.2f}s)")

    # --- With TurboQuant K4/V2 ---
    print(f"\nGenerating with TurboQuant (K4/V2, residual_window=128)...")
    wavs, sr = model.generate_custom_voice(
        text=text,
        language=language,
        speaker=speaker,
        turboquant_config=TurboQuantConfig(key_bits=4, value_bits=2),
    )
    sf.write("output_turboquant.wav", wavs[0], sr)
    print(f"  Saved output_turboquant.wav ({len(wavs[0]) / sr:.2f}s)")

    # Print compression stats
    cache = model.model.last_kv_cache
    if hasattr(cache, "memory_report"):
        report = cache.memory_report()
        total_mb = report["total_bytes"] / 1024 ** 2
        fp16_mb = report["fp16_equivalent_bytes"] / 1024 ** 2
        ratio = report["compression_ratio"]
        print(f"\n  KV cache memory: {total_mb:.1f} MB (fp16 equivalent: {fp16_mb:.1f} MB)")
        print(f"  Compression ratio: {ratio:.2f}x")

    # --- With aggressive compression K2/V2, smaller window ---
    print(f"\nGenerating with TurboQuant (K2/V2, residual_window=64)...")
    wavs, sr = model.generate_custom_voice(
        text=text,
        language=language,
        speaker=speaker,
        turboquant_config=TurboQuantConfig(key_bits=2, value_bits=2, residual_window=64),
    )
    sf.write("output_turboquant_aggressive.wav", wavs[0], sr)
    print(f"  Saved output_turboquant_aggressive.wav ({len(wavs[0]) / sr:.2f}s)")

    cache = model.model.last_kv_cache
    if hasattr(cache, "memory_report"):
        report = cache.memory_report()
        total_mb = report["total_bytes"] / 1024 ** 2
        fp16_mb = report["fp16_equivalent_bytes"] / 1024 ** 2
        ratio = report["compression_ratio"]
        print(f"\n  KV cache memory: {total_mb:.1f} MB (fp16 equivalent: {fp16_mb:.1f} MB)")
        print(f"  Compression ratio: {ratio:.2f}x")

    print("\nDone. Listen to the output files to compare quality.")


if __name__ == "__main__":
    main()
