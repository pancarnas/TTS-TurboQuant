"""KV + attention-map divergence experiment across a set of (bits, rw) configs.

Phase 1 (counterfactual, deterministic). Per sentence it (1) generates and saves
the compressed audio under each requested config, and (2) runs ONE fp16 pass
during which it measures, per layer/decode-position/config, how much compression
moves the talker's attention map and KV vectors:

  * attention: Jensen-Shannon divergence between softmax(q.Kfp16) and
    softmax(q.Kcompressed) on the SAME fp16 query (base-2, bounded [0, 1]);
    plus total-variation, top-1 agreement, entropy delta, attention-output cosine.
    Both maps are recomputed in fp32 through the same code path, so identical
    inputs give exactly 0 divergence (no bf16-rounding noise floor);
  * KV: per-vector cosine + relative MSE of compressed-vs-fp16 keys/values,
    computed on the COMPRESSED PREFIX only (the exact fp16 residual tail is
    excluded, so rw comparisons measure per-token error, not window coverage).

Rows carry a ``protected_layers`` column and wavs a ``_pl=<n>`` name segment, so
runs with different ``--protected-layers`` never conflate (also keys --resume).
Likewise the key-fix flags (docs/keyfix_experiment.md): ``--key-group N``
(per-group min-max key scaling, OmniKVQuant-style) adds a ``key_group`` column
and a ``_g=<N>`` wav segment; ``--pre-rope-keys`` (quantize keys before RoPE)
adds a ``pre_rope`` column and a ``_pre`` wav segment. Both are run-wide, like
``--protected-layers``: one invocation = one experimental cell.

Configs are given as ``--configs "K4V4@24,K4V3@24,K4V2@24,K3V3@24,K4V4@0"``
(K<key_bits>V<value_bits>@<residual_window>). The special token ``fp16`` is
uncompressed inference (config=None, no TQ hooks): its wav is the quality
baseline for offline CER/speaker-sim scoring, it contributes no divergence rows
(fp16 vs fp16 is zero by construction), and its filename carries no _pl segment
since protection doesn't apply. No Whisper/ASR, no WavLM — run the
whisper pipeline separately on the saved wavs. Divergence rows STREAM to the CSV
per sentence (bounded RAM, crash-safe at sentence granularity).

Run from the repo root (same env as the benchmark; GPU recommended):
  python models/Qwen3-TTS/benchmarks/kv_attn_divergence_experiment.py \
      --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --data-dir data \
      --groups seedtts_en,librispeech_pc,libritts_long,ellav_hard \
      --max-per-group 100 --configs "K4V4@24,K4V3@24,K4V2@24,K3V3@24,K4V4@0" \
      --step-stride 1 --out results/kv_attn.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import traceback

import pandas as pd
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import qwen_tts.core.models.modeling_qwen3_tts as modeling  # noqa: E402
from qwen_tts.core.models.turboquant_kv_cache import rope_keys  # noqa: E402

from benchmark_qwen3tts_real import (  # noqa: E402 - needs sys.path tweak above
    Qwen3TTSModel,
    _kv_recon_errors,
    _resolve_voice,
    run_generation,
)

from turboquant.bench_common import decode_overrides, set_global_seed  # noqa: E402
from turboquant.compressors_v3 import TurboQuantV3  # noqa: E402
from turboquant.config import TurboQuantConfig  # noqa: E402
from turboquant.eval_sentences import iter_eval_items  # noqa: E402

COLUMNS = [
    "group",
    "idx",
    "layer",
    "pos",
    "key_bits",
    "value_bits",
    "rw",
    "protected_layers",
    "key_group",
    "pre_rope",
    "protect_prefix",
    "prefix_codebook",
    "prefix_bits",
    "attn_js",
    "attn_tv",
    "attn_top1",
    "attn_dentropy",
    "out_cos",
    "cos_k",
    "cos_v",
    "relmse_k",
    "relmse_v",
]
_N_ID = 13  # id/config columns before the metrics
_METRICS = COLUMNS[_N_ID:]

_CFG_RE = re.compile(r"[Kk](\d+)[Vv](\d+)@(\d+)$")

# --configs token "fp16": uncompressed generation (config=None), audio baseline.
FP16_SPEC = (None, None, None)


def parse_configs(spec: str) -> list[tuple]:
    """'K4V4@24,fp16' -> [(4,4,24), FP16_SPEC]  ((key_bits, value_bits, rw))."""
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok.lower() == "fp16":
            out.append(FP16_SPEC)
            continue
        m = _CFG_RE.match(tok)
        if not m:
            raise SystemExit(f"bad config {tok!r}; expected e.g. K4V4@24 or fp16")
        out.append((int(m.group(1)), int(m.group(2)), int(m.group(3))))
    return out


def config_label(kb, vb, rw) -> str:
    return "fp16" if kb is None else f"K{kb}V{vb}@{rw}"


def make_config(
    kb: int,
    vb: int,
    rw: int,
    protected_layers: int = 2,
    key_group: int = 0,
    pre_rope_keys: bool = False,
    protect_prefix: int = 0,
    prefix_codebook: str = "",
    prefix_bits: int = 0,
) -> TurboQuantConfig:
    """A config with REAL on-path compression (track_only=False).

    Default track_only=True makes attention read pristine fp16 KV — generation
    then ignores compression and every config yields identical audio. Force False.

    ``protected_layers`` = first/last N layers kept at protected_bits; 0 disables
    layer-adaptive protection (uniform bits everywhere, matching the reference impl).
    ``key_group`` / ``pre_rope_keys``: the key-fix experiment flags (see module doc).
    """
    return TurboQuantConfig(
        key_bits=kb,
        value_bits=vb,
        residual_window=rw,
        protected_layers=protected_layers,
        track_only=False,
        key_group=key_group,
        pre_rope_keys=pre_rope_keys,
        protect_prefix=protect_prefix,
        prefix_codebook=prefix_codebook,
        prefix_bits=prefix_bits,
    )


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> float:
    """Mean Jensen-Shannon divergence (base-2, in [0, 1]) over the last-dim dists."""
    p = p.float().clamp_min(eps)
    q = q.float().clamp_min(eps)
    m = 0.5 * (p + q)
    kl_pm = (p * (p / m).log2()).sum(-1)
    kl_qm = (q * (q / m).log2()).sum(-1)
    return float((0.5 * kl_pm + 0.5 * kl_qm).mean().item())


def total_variation(p: torch.Tensor, q: torch.Tensor) -> float:
    """Mean total-variation distance (fraction of attention mass moved, in [0, 1])."""
    return float((0.5 * (p.float() - q.float()).abs().sum(-1)).mean().item())


def top1_agreement(p: torch.Tensor, q: torch.Tensor) -> float:
    """Fraction of queries whose most-attended key position is unchanged."""
    return float((p.argmax(-1) == q.argmax(-1)).float().mean().item())


def entropy_delta(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> float:
    """Mean H(q) - H(p) (base-2): positive => compression smears attention wider."""

    def _h(x):
        x = x.float().clamp_min(eps)
        return -(x * x.log2()).sum(-1)

    return float((_h(q) - _h(p)).mean().item())


def output_cosine(o_fp16: torch.Tensor, o_comp: torch.Tensor) -> float:
    """Mean cosine between the attention-output vectors (what feeds the next layer)."""
    a = o_fp16.reshape(-1, o_fp16.shape[-1]).float()
    b = o_comp.reshape(-1, o_comp.shape[-1]).float()
    return float(torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item())


def compressed_attention(query, rk, num_kv_groups, attention_mask, scaling):
    """Recompute the eager attention map using reconstructed keys (matches model)."""
    ks = modeling.repeat_kv(rk, num_kv_groups)
    aw = torch.matmul(query, ks.transpose(2, 3)) * scaling
    if attention_mask is not None:
        aw = aw + attention_mask[:, :, :, : ks.shape[-2]]
    return torch.nn.functional.softmax(aw, dim=-1, dtype=torch.float32)


class DivergenceRecorder:
    """Per-(layer, pos, config) attention + KV divergence during one fp16 pass."""

    def __init__(
        self,
        specs,
        n_layers,
        prot_layers,
        prot_bits,
        stride,
        key_group: int = 0,
        pre_rope: bool = False,
        protect_prefix: int = 0,
        prefix_codebook: str = "",
        prefix_bits: int = 0,
    ):
        self.specs = specs  # list of (key_bits, value_bits, rw)
        self.n_layers, self.pl, self.pb, self.stride = (
            n_layers,
            prot_layers,
            prot_bits,
            stride,
        )
        self.key_group = int(key_group)
        self.pre_rope = bool(pre_rope)
        # protect_prefix: -1 = the prefill length of each attention class
        # (learned on the prefill call), N > 0 = fixed; the first `prefix`
        # positions are spliced back in fp16 after compress/decompress.
        self.protect_prefix = int(protect_prefix)
        # prefix_codebook: the prompt is quantized (fitted levels) instead of spliced in fp16.
        self.prefix_codebook = prefix_codebook
        self.prefix_bits = int(prefix_bits)
        self._prefix: dict = {}
        self.active = False
        self.rows: list[tuple] = []
        self.errors = 0
        self.first_error: str | None = None
        self.group = self.idx = None
        self._cache: dict = {}
        # pre_rope: per-attention-class running copy of the raw rotary tables
        # (cos, sin) for every position seen so far in the current generation.
        self._rope_buf: dict = {}
        # prompt round trips, reused across steps (the prefill tensor is fixed within a sentence)
        self._prompt_rt: dict = {}

    def _comp(self, layer, kb, vb, rw, head_dim, device):
        key = (layer, kb, vb, rw)
        if key not in self._cache:
            self._cache[key] = TurboQuantV3(
                head_dim=head_dim,
                key_bits=kb,
                value_bits=vb,
                residual_window=rw,
                layer_idx=layer,
                n_layers=self.n_layers,
                protected_layers=self.pl,
                protected_bits=self.pb,
                seed=42,
                device=str(device),
                key_group=self.key_group,
                prefix_codebook=self.prefix_codebook,
                prefix_bits=self.prefix_bits,
            )
        return self._cache[key]

    def _rope_tables(self, module, pos):
        """Raw (cos, sin) for positions [0, pos) of this module's sequence.

        The attention layer stashes this step's tables on ``module._tq_rope``
        (see modeling_qwen3_tts). They are accumulated per attention CLASS
        (talker vs code predictor run different sequences). Every LAYER of a
        class calls this with the same step's tables, so a step is appended
        once (first layer) and reused by the others; a step whose tables
        neither restart, extend nor repeat the buffer is a protocol error.
        """
        cos, sin = module._tq_rope
        name = type(module).__name__
        buf = self._rope_buf.get(name)
        n_new = cos.shape[-2]
        have = 0 if buf is None else buf[0].shape[-2]
        if n_new == pos:  # prefill (or a sequence restart): full tables
            buf = (cos, sin)
        elif have + n_new == pos:  # first layer of a decode step: extend
            buf = (torch.cat([buf[0], cos], dim=-2), torch.cat([buf[1], sin], dim=-2))
        elif have == pos and torch.equal(buf[0][..., -n_new:, :], cos):
            pass  # another layer of the same step: already appended
        else:
            raise RuntimeError(
                f"pre_rope: rotary tables cover {n_new} of {pos} positions for {name} "
                f"(buffer holds {have})"
            )
        self._rope_buf[name] = buf
        return buf

    def _prefix_len(self, module, query, pos):
        """Protected prefix for this module's sequence (0 when off)."""
        if self.protect_prefix == 0:
            return 0
        if self.protect_prefix > 0:
            return min(self.protect_prefix, pos)
        name = type(module).__name__
        if query.shape[2] == pos:  # prefill (or restart): the prompt length
            self._prefix[name] = pos
        return min(self._prefix.get(name, 0), pos)

    def _prompt_round_trip(self, module, comp, key_in, value, pfx, key):
        """Decompressed prompt (keys pre-RoPE, values), memoized per (class, layer, config, pfx)."""
        pk_in, pv_in = key_in[:, :, :pfx], value[:, :, :pfx]
        hit = self._prompt_rt.get(key)
        if (
            hit is not None
            and torch.equal(pk_in, hit[0])
            and torch.equal(pv_in, hit[1])
        ):
            return hit[2], hit[3]
        ck, cv = comp.compress_kv(pk_in, pv_in, prefix=pfx)
        rpk, rpv = comp.decompress_kv(ck, cv)
        self._prompt_rt[key] = (pk_in.clone(), pv_in.clone(), rpk, rpv)
        return rpk, rpv

    def _splice_quantized_prefix(
        self, module, comp, key_in, value, rk, rv, pfx, rope, dtype, spec
    ):
        """Replace the first ``pfx`` reconstructed positions with the prefix-codebook round trip."""
        cache_key = (type(module).__name__, module.layer_idx, *spec, pfx)
        rpk, rpv = self._prompt_round_trip(module, comp, key_in, value, pfx, cache_key)
        if rope is not None:
            head = {
                **rope,
                "cos": rope["cos"][..., :pfx, :],
                "sin": rope["sin"][..., :pfx, :],
            }
            rpk = rope_keys(rpk.to(dtype), inverse=False, **head)
        rk = torch.cat([rpk.to(rk.dtype), rk[:, :, pfx:]], dim=2)
        rv = torch.cat([rpv.to(rv.dtype), rv[:, :, pfx:]], dim=2)
        return rk, rv

    def record(self, module, query, key, value, attention_mask, scaling):
        pos = key.shape[2]
        if self.pre_rope:  # must see EVERY step, even the ones the stride skips
            cos, sin = self._rope_tables(module, pos)
        pfx = self._prefix_len(module, query, pos)  # must see every step too
        if self.stride > 1 and (pos % self.stride) != 0:
            return
        head_dim = key.shape[-1]
        groups = module.num_key_value_groups
        # fp16 reference through the SAME fp32 code path as the compressed map:
        # identical K then gives exactly 0 divergence (the eager output is cast
        # to bf16 after softmax, which would floor JS/TV at rounding noise).
        af = compressed_attention(query, key, groups, attention_mask, scaling)
        rope = None
        if self.pre_rope:
            rs = getattr(module, "rope_scaling", None) or {}
            rope = dict(
                cos=cos,
                sin=sin,
                mrope_section=rs.get("mrope_section") if cos.dim() == 4 else None,
                interleaved=bool(rs.get("interleaved", False)),
            )
            key_in = rope_keys(key, inverse=True, **rope)
        else:
            key_in = key
        for kb, vb, rw in self.specs:
            if kb is None:  # fp16 baseline: audio-only, zero divergence by construction
                continue
            comp = self._comp(module.layer_idx, kb, vb, rw, head_dim, key.device)
            ck, cv = comp.compress_kv(key_in, value)
            rk, rv = comp.decompress_kv(ck, cv)
            if rope is not None:  # back to post-RoPE space for the attention map
                rk = rope_keys(rk.to(key.dtype), inverse=False, **rope)
            if (
                pfx > 0 and self.prefix_codebook
            ):  # prompt quantized with its own codebook
                rk, rv = self._splice_quantized_prefix(
                    module,
                    comp,
                    key_in,
                    value,
                    rk,
                    rv,
                    pfx,
                    rope,
                    key.dtype,
                    (kb, vb, rw),
                )
            elif pfx > 0:  # protected prefix: exact fp16 keys AND values
                rk = torch.cat([key[:, :, :pfx].to(rk.dtype), rk[:, :, pfx:]], dim=2)
                rv = torch.cat([value[:, :, :pfx].to(rv.dtype), rv[:, :, pfx:]], dim=2)
            lo = (
                0 if self.prefix_codebook else pfx
            )  # quantized prompt counts as compressed
            ac = compressed_attention(query, rk, groups, attention_mask, scaling)
            o_fp16 = torch.matmul(af, modeling.repeat_kv(value, groups).float())
            o_comp = torch.matmul(ac, modeling.repeat_kv(rv, groups).float())
            # KV recon on the compressed prefix only — the last rw tokens are
            # bit-exact fp16 and would dilute the error as S grows past rw.
            split_at = max(pos - rw, 0)
            if split_at <= lo:  # nothing compressed at this position
                cos_k = cos_v = 1.0
                relmse_k = relmse_v = 0.0
            else:
                cos_k, relmse_k = _kv_recon_errors(
                    key[:, :, lo:split_at], rk[:, :, lo:split_at], head_dim
                )
                cos_v, relmse_v = _kv_recon_errors(
                    value[:, :, lo:split_at], rv[:, :, lo:split_at], head_dim
                )
            self.rows.append(
                (
                    self.group,
                    self.idx,
                    module.layer_idx,
                    pos,
                    kb,
                    vb,
                    rw,
                    self.pl,
                    self.key_group,
                    int(self.pre_rope),
                    self.protect_prefix,
                    self.prefix_codebook,
                    self.prefix_bits,
                    js_divergence(af, ac),
                    total_variation(af, ac),
                    top1_agreement(af, ac),
                    entropy_delta(af, ac),
                    output_cosine(o_fp16, o_comp),
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
                recorder.record(module, query, key, value, attention_mask, scaling)
            except Exception:  # noqa: BLE001 - a measurement error must not kill the run
                recorder.errors += 1
                if recorder.first_error is None:
                    recorder.first_error = traceback.format_exc()
        return out, attn

    return patched


def resolve_n_layers(model) -> int:
    """Talker layer count for the protected-layer logic — fail loudly if unresolved.

    The cache uses ``config.talker_config.num_hidden_layers``. The TOP config has
    no bare ``num_hidden_layers``, so ``getattr(model.model.config,
    'num_hidden_layers', 0)`` returned 0 -> TurboQuantV3 marked EVERY layer as
    protected (8-bit) -> key/value bits were silently ignored (all configs
    identical). Resolve the talker count and refuse to run on 0.
    """
    for getter in (
        lambda: model.model.config.talker_config.num_hidden_layers,
        lambda: model.config.talker_config.num_hidden_layers,
        lambda: model.model.config.num_hidden_layers,
    ):
        try:
            n = int(getter())
        except (AttributeError, TypeError):
            continue
        if n > 0:
            return n
    raise SystemExit(
        "could not resolve talker num_hidden_layers (>0); refusing to run — the "
        "protected-layer logic would force every layer to 8-bit and ignore the "
        "requested key/value bits"
    )


def force_eager(model) -> None:
    """Set every attention config to eager so the patched path is used.

    ``Qwen3TTSModel`` is a wrapper, not an ``nn.Module`` — the talker lives at
    ``model.model`` — so walk the real sub-modules rather than ``model`` itself.
    """
    candidates = [model, getattr(model, "model", None)]
    candidates += [v for v in vars(model).values() if isinstance(v, torch.nn.Module)]
    for mod in candidates:
        if not isinstance(mod, torch.nn.Module):
            continue
        for m in mod.modules():
            cfg = getattr(m, "config", None)
            if cfg is not None and hasattr(cfg, "_attn_implementation"):
                cfg._attn_implementation = "eager"


def accumulate_running(running: dict, rows: list) -> None:
    """Fold streamed rows into per-config running sums (bounded memory)."""
    for r in rows:
        cfg = (
            r[4],
            r[5],
            r[6],
        )  # (key_bits, value_bits, rw); r[7..10] = pl, key_group, pre_rope, protect_prefix
        d = running.setdefault(cfg, dict({"_n": 0}, **{m: 0.0 for m in _METRICS}))
        d["_n"] += 1
        for m, v in zip(_METRICS, r[_N_ID:]):
            d[m] += float(v)


def running_frame(running: dict) -> pd.DataFrame:
    """Per-config means + Δ(0-24) rows for any bits present at both rw 0 and 24."""
    means = {
        config_label(*cfg): {m: d[m] / d["_n"] for m in _METRICS}
        for cfg, d in running.items()
        if d["_n"]
    }
    agg = pd.DataFrame(means).T.sort_index()
    bits_rws: dict = {}
    for kb, vb, rw in running:
        bits_rws.setdefault((kb, vb), set()).add(rw)
    for (kb, vb), rws in bits_rws.items():
        if {0, 24} <= rws:
            lo, hi = config_label(kb, vb, 0), config_label(kb, vb, 24)
            agg.loc[f"K{kb}V{vb} d(0-24)"] = agg.loc[lo] - agg.loc[hi]
    return agg


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Offline: per-config means from a streamed CSV."""
    keys = ["key_bits", "value_bits", "rw", "protected_layers"]
    keys += [c for c in ("key_group", "pre_rope", "protect_prefix") if c in df.columns]
    return df.groupby(keys)[_METRICS].mean().sort_index()


def _wav_path(
    out_dir,
    group,
    idx,
    seed,
    temp,
    kb,
    vb,
    rw,
    pl,
    key_group=0,
    pre_rope=False,
    protect_prefix=0,
    prefix_codebook="",
    prefix_bits=0,
):
    # fp16 carries no _pl/_g/_pre/_pp segment: none of them apply to uncompressed
    # generation, so runs at different settings share one baseline wav.
    if kb is None:
        cfg = "fp16"
    else:
        cfg = f"K{kb}_V{vb}_rw={rw}_pl={pl}"
        if key_group:
            cfg += f"_g={key_group}"
        if pre_rope:
            cfg += "_pre"
        if prefix_codebook:  # replaces _pp: the prompt is quantized, not fp16
            cfg += f"_pq={prefix_codebook[0]}{prefix_bits or kb}"
        elif protect_prefix:
            cfg += "_pp=prompt" if protect_prefix < 0 else f"_pp={protect_prefix}"
    return os.path.join(
        out_dir, f"qwen_{group}_{idx}_sampling_s{seed}_t{temp}_{cfg}.wav"
    )


def load_done(
    out_path: str,
    protected_layers: int,
    key_group: int = 0,
    pre_rope: bool = False,
    protect_prefix: int = 0,
    prefix_codebook: str = "",
    prefix_bits: int = 0,
) -> set:
    """(group, idx) pairs already recorded in a divergence CSV (for --resume).

    Only rows recorded under the SAME ``protected_layers`` (and, when the CSV
    has the columns, the same ``key_group`` / ``pre_rope``) count as done — a
    CSV can hold runs at different settings without --resume skipping the wrong
    ones. Reads only the id columns, so it stays cheap even on multi-GB CSVs.
    A missing/empty file (or one with only a header) yields an empty set.
    """
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        return set()
    try:
        header = pd.read_csv(out_path, nrows=0).columns
        opt = "key_group pre_rope protect_prefix prefix_codebook prefix_bits".split()
        cols = ["group", "idx", "protected_layers"] + [c for c in opt if c in header]
        df = pd.read_csv(out_path, usecols=cols)
    except ValueError as exc:
        raise SystemExit(
            f"{out_path} has no protected_layers column (pre-fix CSV) — cannot "
            "--resume into it; start a fresh --out"
        ) from exc
    df = df[df["protected_layers"] == protected_layers]
    if "key_group" in df.columns:
        df = df[df["key_group"] == key_group]
    if "pre_rope" in df.columns:
        df = df[df["pre_rope"] == int(pre_rope)]
    if "protect_prefix" in df.columns:
        df = df[df["protect_prefix"] == int(protect_prefix)]
    elif protect_prefix:  # pre-flag CSV holds no protected rows
        return set()
    if "prefix_codebook" in df.columns:
        df = df[df["prefix_codebook"].fillna("") == prefix_codebook]
        df = df[df["prefix_bits"] == int(prefix_bits)]
    elif prefix_codebook:  # pre-flag CSV holds no codebook rows
        return set()
    return set(zip(df["group"].astype(str), df["idx"].astype(int)))


def pending_items(items: list, done: set) -> list:
    """Drop (group, idx, item) tuples whose (group, idx) is already done."""
    return [t for t in items if (str(t[0]), int(t[1])) not in done]


def load_idx_file(path):
    """One integer idx per line ('#' comments and blanks ok) -> set; None if no path."""
    if not path:
        return None
    allowed = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                allowed.add(int(line))
    return allowed


def select_idxs(items: list, allowed) -> list:
    """Keep (group, idx, item) tuples whose idx is in ``allowed`` (None = keep all)."""
    return items if allowed is None else [t for t in items if t[1] in allowed]


def wav_present(path: str) -> bool:
    """True iff the wav exists and is non-empty (per-config skip in audio-only resume)."""
    return os.path.exists(path) and os.path.getsize(path) > 0


def audio_done(
    out_dir,
    group,
    idx,
    seed,
    temp,
    specs,
    pl,
    key_group=0,
    pre_rope=False,
    protect_prefix=0,
    prefix_codebook="",
    prefix_bits=0,
) -> bool:
    """True iff every config's wav for this sentence already exists (audio resume)."""
    return all(
        os.path.exists(
            _wav_path(
                out_dir,
                group,
                idx,
                seed,
                temp,
                kb,
                vb,
                rw,
                pl,
                key_group,
                pre_rope,
                protect_prefix,
                prefix_codebook,
                prefix_bits,
            )
        )
        for kb, vb, rw in specs
    )


def check_resume_header(out_path: str) -> None:
    """Refuse to append to a CSV written with a different column set (older runs have fewer id columns)."""
    have = list(pd.read_csv(out_path, nrows=0).columns)
    if have != COLUMNS:
        raise SystemExit(
            f"--resume: {out_path} has a different header than this branch writes "
            f"({len(have)} vs {len(COLUMNS)} columns); use a fresh --out instead of appending"
        )


def run_experiment(model, speaker, recorder, args) -> dict:
    """Generate audio + (unless --no-divergence) stream divergence rows to args.out.

    Rows are written and freed per sentence (bounded RAM, crash-safe at sentence
    granularity). Returns the per-config running aggregate for the summary.
    """
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    specs = recorder.specs
    out_dir = getattr(args, "audio_out_dir", None) or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "outputs"
    )
    os.makedirs(out_dir, exist_ok=True)
    allowed = load_idx_file(getattr(args, "idx_file", None))
    max_per_group = None if allowed is not None else args.max_per_group
    items = select_idxs(
        [
            (g, i, it)
            for g in groups
            for i, it in enumerate(iter_eval_items([g], max_per_group, args.data_dir))
        ],
        allowed,
    )
    if allowed is not None:
        print(f"restricting to {len(items)} idxs from {args.idx_file}")

    no_div = getattr(args, "no_divergence", False)
    resume = getattr(args, "resume", False)
    if resume:  # drop sentences already finished (skip audio + divergence)
        n0 = len(items)
        if no_div:
            items = [
                t
                for t in items
                if not audio_done(
                    out_dir,
                    t[0],
                    t[1],
                    args.seed,
                    args.temperature,
                    specs,
                    args.protected_layers,
                    args.key_group,
                    args.pre_rope_keys,
                    args.protect_prefix,
                    args.prefix_codebook,
                    args.prefix_bits,
                )
            ]
        else:
            items = pending_items(
                items,
                load_done(
                    args.out,
                    args.protected_layers,
                    args.key_group,
                    args.pre_rope_keys,
                    args.protect_prefix,
                    args.prefix_codebook,
                    args.prefix_bits,
                ),
            )
        print(f"resume: skipping {n0 - len(items)} done, running {len(items)}")

    fh = writer = None
    running: dict = {}
    if not no_div:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        append = resume and os.path.exists(args.out) and os.path.getsize(args.out) > 0
        if append:
            check_resume_header(args.out)
        fh = open(args.out, "a" if append else "w", newline="", encoding="utf-8")
        writer = csv.writer(fh)
        if not append:  # fresh file gets a header; append continues an existing one
            writer.writerow(COLUMNS)

    it = tqdm(items, desc="kv-attn", unit="sent") if tqdm else items
    try:
        for group, idx, item in it:
            ref_audio, ref_text = _resolve_voice(
                item, args.voice_mode, args.default_ref, None
            )
            if args.voice_mode == "clone" and not ref_audio:
                continue
            for kb, vb, rw in specs:  # per-config audio (recorder off); fp16 -> no TQ
                wav_out = _wav_path(
                    out_dir,
                    group,
                    idx,
                    args.seed,
                    args.temperature,
                    kb,
                    vb,
                    rw,
                    args.protected_layers,
                    args.key_group,
                    args.pre_rope_keys,
                    args.protect_prefix,
                    args.prefix_codebook,
                    args.prefix_bits,
                )
                if resume and no_div and wav_present(wav_out):
                    continue  # this config's wav exists: fill only the missing ones
                recorder.active = False
                set_global_seed(args.seed, deterministic=False)
                wavs, sr, *_ = run_generation(
                    model,
                    item.text,
                    "English",
                    speaker,
                    None
                    if kb is None
                    else make_config(
                        kb,
                        vb,
                        rw,
                        args.protected_layers,
                        args.key_group,
                        args.pre_rope_keys,
                        args.protect_prefix,
                        args.prefix_codebook,
                        args.prefix_bits,
                    ),
                    seed=args.seed,
                    gen_overrides=decode_overrides(
                        "sampling", temperature=args.temperature
                    ),
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                )
                sf.write(wav_out, wavs[0], sr)
            if no_div:
                continue  # audio-only: skip the expensive fp16 recording pass
            recorder.group, recorder.idx = group, idx  # fp16 pass (recorder on)
            recorder._prompt_rt.clear()
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
            if writer and recorder.rows:  # stream + free per sentence
                writer.writerows(recorder.rows)
                accumulate_running(running, recorder.rows)
                recorder.rows.clear()
                fh.flush()
    finally:
        if fh:
            fh.close()
    if recorder.errors:
        print(f"  WARNING: {recorder.errors} measurement errors (skipped)")
        if recorder.first_error:
            print(f"  first error:\n{recorder.first_error}")
    return running


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--groups", default="seedtts_en,librispeech_pc,libritts_long,ellav_hard"
    )
    parser.add_argument("--max-per-group", type=int, default=100)
    parser.add_argument(
        "--configs",
        default="K4V4@24,K4V3@24,K4V2@24,K3V3@24,K4V4@0",
        help="Comma list of K<kb>V<vb>@<rw> (e.g. K4V4@24,K4V4@0). The token 'fp16' "
        "adds an uncompressed baseline: audio only, no divergence rows.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip sentences already in --out (or, with --no-divergence, whose wavs "
        "exist) and append instead of overwriting. Use only on CSVs from this fix.",
    )
    parser.add_argument(
        "--protected-layers",
        type=int,
        default=TurboQuantConfig().protected_layers,
        help="First/last N layers kept at protected_bits (8-bit). 0 = no layer-adaptive "
        "protection: uniform key/value bits on every layer, matching the reference "
        "implementation. Default keeps the current behavior.",
    )
    parser.add_argument(
        "--key-group",
        type=int,
        default=0,
        help="Key-fix E1: per-group min-max scaling of rotated keys over N consecutive "
        "tokens (OmniKVQuant temporal range adaptation; paper uses 32). 0 = off. "
        "Run-wide: wavs get a _g=<N> segment, rows a key_group column.",
    )
    parser.add_argument(
        "--pre-rope-keys",
        action="store_true",
        help="Key-fix E2: quantize keys BEFORE rotary position embedding (RoPE "
        "re-applied to the reconstruction). Run-wide: wavs get a _pre segment, "
        "rows a pre_rope column.",
    )
    parser.add_argument(
        "--protect-prefix",
        type=int,
        default=0,
        help="Keep the first N cache positions (keys and values) in fp16: -1 = the "
        "whole prefill (the prompt), N > 0 = fixed count, 0 = off. Run-wide: wavs "
        "get a _pp=prompt / _pp=<N> segment, rows a protect_prefix column.",
    )
    parser.add_argument(
        "--prefix-codebook",
        choices=["", "kmeans", "minmax"],
        default="",
        help="Quantize the prompt (needs --protect-prefix -1) with a fitted-levels "
        "codebook instead of keeping it fp16. Run-wide: wavs get a _pq=k<bits>/m<bits> "
        "segment, rows prefix_codebook/prefix_bits columns.",
    )
    parser.add_argument(
        "--prefix-bits",
        type=int,
        default=0,
        help="Bits for the prompt codebook; 0 = follow each config's key bits.",
    )
    parser.add_argument("--step-stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--voice-mode", default="clone")
    parser.add_argument("--default-ref", default=None)
    parser.add_argument("--out", default="results/kv_attn.csv")
    parser.add_argument(
        "--idx-file",
        default=None,
        help="Restrict to the sentence idxs listed in this file (one per line, # "
        "comments). The whole group is enumerated; --max-per-group is ignored.",
    )
    parser.add_argument(
        "--no-divergence",
        action="store_true",
        help="Audio-only: generate the compressed wavs, skip the fp16 recording pass.",
    )
    parser.add_argument(
        "--audio-out-dir",
        default=None,
        help="Directory for the wavs (default: benchmarks/outputs). Use a separate "
        "dir to run in parallel with another job without filename collisions.",
    )
    args = parser.parse_args()

    specs = parse_configs(args.configs)
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(args.dtype, torch.bfloat16)
    print(f"loading {args.model} on {args.device} ({args.dtype})")
    print(f"configs: {[config_label(*s) for s in specs]}")
    print(
        f"protected_layers: {args.protected_layers}"
        + (" (no layer-adaptive protection)" if args.protected_layers == 0 else "")
    )
    if args.key_group < 0:
        raise SystemExit(f"--key-group must be >= 0, got {args.key_group}")
    if args.prefix_codebook and args.protect_prefix != -1:
        raise SystemExit("--prefix-codebook requires --protect-prefix -1")
    if args.prefix_bits and not args.prefix_codebook:
        raise SystemExit("--prefix-bits requires --prefix-codebook")
    if not 0 <= args.prefix_bits <= 8:
        raise SystemExit(f"--prefix-bits must be in 0..8, got {args.prefix_bits}")
    print(
        f"key-fix: key_group={args.key_group} "
        f"({'group min-max key scaling' if args.key_group else 'fixed codebook'}), "
        f"pre_rope_keys={args.pre_rope_keys}, protect_prefix={args.protect_prefix}, "
        f"prefix_codebook={args.prefix_codebook!r}, prefix_bits={args.prefix_bits}"
    )
    model = Qwen3TTSModel.from_pretrained(
        args.model, device_map=args.device, dtype=dtype
    )
    speakers = model.get_supported_speakers()
    speaker = speakers[0] if speakers else "Ryan"

    n_layers = resolve_n_layers(model)
    print(f"talker n_layers = {n_layers}")
    d = TurboQuantConfig()  # for protected_bits (8-bit); protected_layers from CLI
    recorder = DivergenceRecorder(
        specs,
        n_layers,
        args.protected_layers,
        d.protected_bits,
        args.step_stride,
        key_group=args.key_group,
        pre_rope=args.pre_rope_keys,
        protect_prefix=args.protect_prefix,
        prefix_codebook=args.prefix_codebook,
        prefix_bits=args.prefix_bits,
    )

    if args.no_divergence:  # audio-only: no eager-forcing, no patch, no recording
        print("audio-only mode (--no-divergence): generating compressed wavs only")
        run_experiment(model, speaker, recorder, args)
        print("\ndone — wavs written")
        return

    force_eager(model)
    original = modeling.eager_attention_forward
    modeling.eager_attention_forward = make_patch(original, recorder)
    try:
        running = run_experiment(model, speaker, recorder, args)  # streams to args.out
    finally:
        modeling.eager_attention_forward = original

    print(f"\nstreamed per-(layer,pos,config) rows -> {args.out}")
    print("\n== mean divergence by config ==")
    print(running_frame(running).to_string(float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
