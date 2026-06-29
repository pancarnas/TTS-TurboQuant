"""KV + attention-map divergence experiment: K4/V4 at residual_window 24 vs 0.

Phase 1 (counterfactual, deterministic). Per sentence it (1) generates and saves
the compressed audio under K4/V4 at each residual window, and (2) runs ONE fp16
pass during which it measures, per layer/decode-position, how much compression
moves the talker's attention map and KV vectors:

  * attention: Jensen-Shannon divergence between softmax(q.Kfp16) and
    softmax(q.Kcompressed) on the SAME fp16 query (base-2, bounded [0, 1]);
  * KV: per-vector cosine + relative MSE of compressed-vs-fp16 keys/values.

No Whisper/ASR and no WavLM — run the whisper pipeline separately on the saved
wavs. The fp16 trajectory (where divergence is measured) is consistent with but
not identical to the compressed trajectory that produced the audio; Phase 2
(on-path dual-cache) closes that gap.

Run from the repo root (same env as the benchmark; GPU recommended):
  python models/Qwen3-TTS/benchmarks/kv_attn_divergence_experiment.py \
      --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --data-dir data \
      --groups seedtts_en,librispeech_pc,libritts_long,ellav_hard \
      --max-per-group 100 --residual-windows 24,0 --step-stride 4 \
      --out results/kv_attn_k4v4_rw24_vs_rw0.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import qwen_tts.core.models.modeling_qwen3_tts as modeling  # noqa: E402

from benchmark_qwen3tts_real import (  # noqa: E402 - needs sys.path tweak above
    Qwen3TTSModel,
    _kv_recon_errors,
    _resolve_voice,
    build_turboquant_configs,
    run_generation,
)

from turboquant.bench_common import decode_overrides, set_global_seed  # noqa: E402
from turboquant.compressors_v3 import TurboQuantV3  # noqa: E402
from turboquant.eval_sentences import iter_eval_items  # noqa: E402

COLUMNS = [
    "group",
    "idx",
    "layer",
    "pos",
    "rw",
    "attn_js",
    "cos_k",
    "cos_v",
    "relmse_k",
    "relmse_v",
]


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> float:
    """Mean Jensen-Shannon divergence (base-2, in [0, 1]) over the last-dim dists."""
    p = p.float().clamp_min(eps)
    q = q.float().clamp_min(eps)
    m = 0.5 * (p + q)
    kl_pm = (p * (p / m).log2()).sum(-1)
    kl_qm = (q * (q / m).log2()).sum(-1)
    return float((0.5 * kl_pm + 0.5 * kl_qm).mean().item())


def compressed_attention(query, rk, num_kv_groups, attention_mask, scaling):
    """Recompute the eager attention map using reconstructed keys (matches model)."""
    ks = modeling.repeat_kv(rk, num_kv_groups)
    aw = torch.matmul(query, ks.transpose(2, 3)) * scaling
    if attention_mask is not None:
        aw = aw + attention_mask[:, :, :, : ks.shape[-2]]
    return torch.nn.functional.softmax(aw, dim=-1, dtype=torch.float32)


def k4v4_config(rw: int):
    """The exact K4/V4 config (incl. protected-layer settings) at a residual window."""
    for name, cfg in build_turboquant_configs(rw):
        if cfg is not None and cfg.key_bits == 4 and cfg.value_bits == 4:
            return name, cfg
    raise SystemExit("no K4/V4 config in build_turboquant_configs")


class DivergenceRecorder:
    """Accumulates per-(layer, pos, rw) attention-JS + KV errors during an fp16 pass."""

    def __init__(self, rws, kb, vb, n_layers, prot_layers, prot_bits, stride):
        self.rws, self.kb, self.vb = rws, kb, vb
        self.n_layers, self.pl, self.pb, self.stride = (
            n_layers,
            prot_layers,
            prot_bits,
            stride,
        )
        self.active = False
        self.rows: list[tuple] = []
        self.errors = 0
        self.group = self.idx = None
        self._cache: dict = {}

    def _comp(self, layer, rw, head_dim, device):
        key = (layer, rw)
        if key not in self._cache:
            self._cache[key] = TurboQuantV3(
                head_dim=head_dim,
                key_bits=self.kb,
                value_bits=self.vb,
                residual_window=rw,
                layer_idx=layer,
                n_layers=self.n_layers,
                protected_layers=self.pl,
                protected_bits=self.pb,
                seed=42,
                device=str(device),
            )
        return self._cache[key]

    def record(self, module, query, key, value, attention_mask, scaling, attn_fp16):
        pos = key.shape[2]
        if self.stride > 1 and (pos % self.stride) != 0:
            return
        head_dim = key.shape[-1]
        af = attn_fp16.float()
        for rw in self.rws:
            comp = self._comp(module.layer_idx, rw, head_dim, key.device)
            ck, cv = comp.compress_kv(key, value)
            rk, rv = comp.decompress_kv(ck, cv)
            ac = compressed_attention(
                query, rk, module.num_key_value_groups, attention_mask, scaling
            )
            cos_k, relmse_k = _kv_recon_errors(key, rk, head_dim)
            cos_v, relmse_v = _kv_recon_errors(value, rv, head_dim)
            self.rows.append(
                (
                    self.group,
                    self.idx,
                    module.layer_idx,
                    pos,
                    rw,
                    js_divergence(af, ac),
                    cos_k,
                    cos_v,
                    relmse_k,
                    relmse_v,
                )
            )


def make_patch(original, recorder: DivergenceRecorder):
    def patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
        out, attn = original(
            module, query, key, value, attention_mask, scaling, dropout, **kw
        )
        if recorder.active:
            try:
                recorder.record(
                    module, query, key, value, attention_mask, scaling, attn
                )
            except Exception:  # noqa: BLE001 - a measurement error must not kill the run
                recorder.errors += 1
        return out, attn

    return patched


def force_eager(model) -> None:
    for m in model.modules():
        cfg = getattr(m, "config", None)
        if cfg is not None and hasattr(cfg, "_attn_implementation"):
            cfg._attn_implementation = "eager"


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["attn_js", "cos_k", "cos_v", "relmse_k", "relmse_v"]
    agg = df.groupby("rw")[cols].mean().sort_index(ascending=False)
    if 0 in agg.index and len(agg.index) > 1:
        other = max(w for w in agg.index if w != 0)
        agg.loc[f"diff(0-{other})"] = agg.loc[0] - agg.loc[other]
    return agg


def _wav_path(out_dir, group, idx, seed, temp, rw):
    return os.path.join(
        out_dir, f"qwen_{group}_{idx}_sampling_s{seed}_t{temp}_K4_V4_rw={rw}.wav"
    )


def run_experiment(model, speaker, recorder, args) -> pd.DataFrame:
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    rws = recorder.rws
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    items = [
        (g, i, it)
        for g in groups
        for i, it in enumerate(iter_eval_items([g], args.max_per_group, args.data_dir))
    ]
    it = tqdm(items, desc="kv-attn", unit="sent") if tqdm else items
    for group, idx, item in it:
        ref_audio, ref_text = _resolve_voice(
            item, args.voice_mode, args.default_ref, None
        )
        if args.voice_mode == "clone" and not ref_audio:
            continue
        for rw in rws:  # compressed audio (recorder off)
            recorder.active = False
            _, cfg = k4v4_config(rw)
            set_global_seed(args.seed, deterministic=False)
            wavs, sr, *_ = run_generation(
                model,
                item.text,
                "English",
                speaker,
                cfg,
                seed=args.seed,
                gen_overrides=decode_overrides(
                    "sampling", temperature=args.temperature
                ),
                ref_audio=ref_audio,
                ref_text=ref_text,
            )
            sf.write(
                _wav_path(out_dir, group, idx, args.seed, args.temperature, rw),
                wavs[0],
                sr,
            )
        recorder.group, recorder.idx = group, idx  # fp16 pass (recorder on)
        recorder.active = True
        set_global_seed(args.seed, deterministic=False)
        run_generation(
            model,
            item.text,
            "English",
            speaker,
            None,
            seed=args.seed,
            gen_overrides=decode_overrides("greedy"),
            ref_audio=ref_audio,
            ref_text=ref_text,
        )
        recorder.active = False
    if recorder.errors:
        print(f"  WARNING: {recorder.errors} measurement errors (skipped)")
    return pd.DataFrame(recorder.rows, columns=COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--groups", default="seedtts_en,librispeech_pc,libritts_long,ellav_hard"
    )
    parser.add_argument("--max-per-group", type=int, default=100)
    parser.add_argument("--residual-windows", default="24,0")
    parser.add_argument("--step-stride", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--voice-mode", default="clone")
    parser.add_argument("--default-ref", default=None)
    parser.add_argument("--out", default="results/kv_attn_k4v4_rw24_vs_rw0.csv")
    args = parser.parse_args()

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(args.dtype, torch.bfloat16)
    print(f"loading {args.model} on {args.device} ({args.dtype})")
    model = Qwen3TTSModel.from_pretrained(
        args.model, device_map=args.device, dtype=dtype
    )
    force_eager(model)
    speakers = model.get_supported_speakers()
    speaker = speakers[0] if speakers else "Ryan"

    rws = [int(w) for w in args.residual_windows.split(",") if w.strip()]
    _, cfg = k4v4_config(rws[0])
    n_layers = getattr(model.model.config, "num_hidden_layers", 0)
    recorder = DivergenceRecorder(
        rws,
        cfg.key_bits,
        cfg.value_bits,
        n_layers,
        getattr(cfg, "protected_layers", 2),
        getattr(cfg, "protected_bits", 8),
        args.step_stride,
    )

    original = modeling.eager_attention_forward
    modeling.eager_attention_forward = make_patch(original, recorder)
    try:
        df = run_experiment(model, speaker, recorder, args)
    finally:
        modeling.eager_attention_forward = original

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {len(df)} per-(layer,pos,rw) rows -> {args.out}")
    print(f"\n== mean divergence by residual window (K{recorder.kb}/V{recorder.vb}) ==")
    print(summarize(df).to_string(float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
