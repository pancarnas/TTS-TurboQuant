"""Run TurboQuant benchmarks.

Usage:
    python benchmarks/run_all.py [--device cuda] [--real]

    --real: Run with real model weights (downloads if needed).
            Without this flag, runs synthetic tensor benchmarks only.
"""

import subprocess
import sys
import os


def main():
    device = "cuda" if "--device" not in sys.argv else sys.argv[sys.argv.index("--device") + 1]
    real_mode = "--real" in sys.argv

    benchmarks_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(benchmarks_dir)

    if real_mode:
        script = os.path.join(project_root, "models", "Qwen3-TTS", "benchmarks", "benchmark_qwen3tts_real.py")
    else:
        script = os.path.join(project_root, "models", "Qwen3-TTS", "benchmarks", "benchmark_qwen3tts.py")

    subprocess.run([sys.executable, script, "--device", device], check=False)


if __name__ == "__main__":
    main()
