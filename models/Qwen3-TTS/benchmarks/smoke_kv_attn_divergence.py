"""Box smoke test for kv_attn_divergence_experiment (needs GPU + model).

Runs the divergence experiment on a couple of sentences (K4V4 at rw=24 and rw=0)
and asserts the end-to-end correctness CPU unit tests can't reach: the model
loads, the attention monkeypatch fires during real generation, the CSV + audio
are produced, values are in range, and the physical sanity sign holds (rw=0
distorts attention/KV more than rw=24). Prints a PASS/FAIL checklist and exits
non-zero on any failure, so it's safe to gate the full run on it.

Run from the repo root on the box:
  python models/Qwen3-TTS/benchmarks/smoke_kv_attn_divergence.py \
      --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --data-dir data --max-per-group 2
"""

from __future__ import annotations

import argparse
import os
import sys
import types

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kv_attn_divergence_experiment as exp  # noqa: E402


def _check(results, label, ok, detail=""):
    results.append(bool(ok))
    print(
        f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  ({detail})" if detail else "")
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--group", default="seedtts_en")
    p.add_argument("--max-per-group", type=int, default=2)
    p.add_argument("--step-stride", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="results/kv_attn_smoke.csv")
    args_cli = p.parse_args()

    specs = exp.parse_configs("K4V4@24,K4V4@0")
    print(f"loading {args_cli.model} on {args_cli.device} ...")
    model = exp.Qwen3TTSModel.from_pretrained(
        args_cli.model, device_map=args_cli.device, dtype=torch.bfloat16
    )
    exp.force_eager(model)
    speakers = model.get_supported_speakers()
    speaker = speakers[0] if speakers else "Ryan"

    n_layers = getattr(model.model.config, "num_hidden_layers", 0)
    d = exp.TurboQuantConfig()
    rec = exp.DivergenceRecorder(
        specs, n_layers, d.protected_layers, d.protected_bits, args_cli.step_stride
    )
    run_args = types.SimpleNamespace(
        groups=args_cli.group,
        max_per_group=args_cli.max_per_group,
        data_dir=args_cli.data_dir,
        voice_mode="clone",
        default_ref=None,
        seed=0,
        temperature=0.9,
        out=args_cli.out,
        no_divergence=False,
        audio_out_dir=None,
    )

    original = exp.modeling.eager_attention_forward
    exp.modeling.eager_attention_forward = exp.make_patch(original, rec)
    try:
        exp.run_experiment(model, speaker, rec, run_args)  # streams to run_args.out
    finally:
        exp.modeling.eager_attention_forward = original

    df = pd.read_csv(run_args.out) if os.path.exists(run_args.out) else pd.DataFrame()

    print("\n=== smoke checks ===")
    results: list[bool] = []
    # real compression: track_only must be False or the audio ignores compression
    _check(
        results,
        "audio uses REAL compression (track_only=False)",
        exp.make_config(4, 4, 24).track_only is False,
    )
    _check(results, "patch fired (rows recorded)", len(df) > 0, f"{len(df)} rows")
    _check(results, "no measurement errors", rec.errors == 0, f"errors={rec.errors}")
    present = set(map(int, df["rw"].unique())) if len(df) else set()
    _check(results, "both residual windows present", present == {24, 0}, str(present))
    if len(df):
        _check(
            results,
            "attn_js in [0,1]",
            bool(((df.attn_js >= 0) & (df.attn_js <= 1.001)).all()),
        )
        _check(results, "cos_k <= 1", bool((df.cos_k <= 1.001).all()))

    out_dir = os.path.join(os.path.dirname(exp.__file__), "outputs")
    wav_ok = (
        all(
            os.path.exists(exp._wav_path(out_dir, g, int(i), 0, 0.9, 4, 4, rw))
            for (g, i) in (
                {(r.group, r.idx) for r in df.itertuples()} if len(df) else set()
            )
            for rw in (24, 0)
        )
        if len(df)
        else False
    )
    _check(results, "audio wavs saved for both rws", wav_ok)

    if len(df) and present == {24, 0}:
        m = df.groupby("rw")[["attn_js", "relmse_k", "cos_k"]].mean()
        d_js = m.loc[0, "attn_js"] - m.loc[24, "attn_js"]
        d_rm = m.loc[0, "relmse_k"] - m.loc[24, "relmse_k"]
        d_cos = m.loc[24, "cos_k"] - m.loc[0, "cos_k"]
        _check(
            results, "SANITY attn_js(rw0) > attn_js(rw24)", d_js > 0, f"d={d_js:.4g}"
        )
        _check(
            results, "SANITY relmse_k(rw0) > relmse_k(rw24)", d_rm > 0, f"d={d_rm:.4g}"
        )
        _check(results, "SANITY cos_k(rw24) > cos_k(rw0)", d_cos > 0, f"d={d_cos:.4g}")

    passed = all(results)
    print(
        f"\n{'ALL SMOKE CHECKS PASSED' if passed else 'SMOKE FAILED'} "
        f"({sum(results)}/{len(results)})"
    )
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
