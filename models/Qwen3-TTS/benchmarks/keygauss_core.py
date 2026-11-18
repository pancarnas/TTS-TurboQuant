"""Gaussianity and alignment statistics of cached keys/values (model-free core).

OmniKVQuant (arXiv:2609.11582, Table 1) reports, over 32-token windows, how far
rotated keys sit from the N(0, 1/d) target that the TurboQuant codebook assumes:
retained variance, Wasserstein-1 and Kolmogorov-Smirnov distance per coordinate.
This module computes the same statistics plus the ones that decide which fix to
try, on tensors that the runner (key_gauss_stats.py) pulls out of the model:

  window stats   var_w, w1_w, ks_w      per coordinate within g-token windows
  global stats   var_g, offset, drift   whole-sequence variance, energy of the
                                        per-coordinate mean, share of variance
                                        that lives BETWEEN windows (temporal drift)
  quant errors   relmse for the fixed Lloyd-Max codebook, per-window min-max
                 (the paper's fix) and a static per-coordinate affine map
                 calibrated on other sentences (the zero-metadata alternative)
  alignment      effective rank of the query second moment, of the key
                 covariance, and the share of key variance inside the top-k
                 query directions (what a query-aware rotation would exploit)

Everything is in the TurboQuant coordinate system: unit-norm vectors, rotated
by the same random Pi the compressor uses, scaled by sqrt(D) so each coordinate
targets N(0, 1). No model code is imported here, so it is testable on CPU.
"""

from __future__ import annotations

import math
import traceback

import pandas as pd
import torch

from turboquant.fitted_levels import assign, fit_levels, reconstruct
from turboquant.lloyd_max import LloydMaxCodebook
from turboquant.turboquant import generate_rotation_matrix

WINDOW_STATS = ["var_w", "w1_w", "ks_w", "n_win"]
GLOBAL_STATS = ["var_g", "offset", "drift", "n_tok"]


# --------------------------------------------------------------------------
# coordinates
# --------------------------------------------------------------------------


def rotate_unit(x: torch.Tensor, Pi: torch.Tensor) -> torch.Tensor:
    """(S, D) raw vectors -> (S, D) rotated unit vectors scaled by sqrt(D).

    Mirrors MSECompressor.compress (normalize, rotate) and rescales so the
    TurboQuant target per coordinate is N(0, 1) instead of N(0, 1/D).
    """
    x = x.float()
    norms = torch.norm(x, dim=-1, keepdim=True)
    return (x / (norms + 1e-8)) @ Pi.T * math.sqrt(x.shape[-1])


def layer_rotation(
    head_dim: int, layer_idx: int, seed: int = 42, device: str = "cpu"
) -> torch.Tensor:
    """The key rotation TurboQuantV3 would use for this layer (seed + layer_idx*1000)."""
    return generate_rotation_matrix(
        head_dim, seed=seed + layer_idx * 1000, device=device
    )


# --------------------------------------------------------------------------
# Table-1 statistics
# --------------------------------------------------------------------------


def window_stats(z: torch.Tensor, g: int = 32) -> dict:
    """Per-coordinate Gaussianity inside non-overlapping g-token windows.

    z: (S, D) coordinates whose target is N(0, 1). Returns means over all
    (window, coordinate) pairs of: the biased sample variance (a true N(0,1)
    window retains (g-1)/g of it), the W1 distance between the window's
    empirical distribution and N(0, 1) (quantile form), and the KS statistic.
    Windows are whole; the trailing partial window is dropped. n_win counts
    (window, coordinate) pairs so results can be pooled by weight.
    """
    S, D = z.shape
    W = S // g
    if W == 0:
        return dict(var_w=float("nan"), w1_w=float("nan"), ks_w=float("nan"), n_win=0)
    zw = z[: W * g].reshape(W, g, D).float()
    var = zw.var(dim=1, unbiased=False)  # (W, D)
    s, _ = torch.sort(zw, dim=1)  # (W, g, D)
    i = torch.arange(g, device=z.device, dtype=torch.float32)
    q = torch.special.ndtri((i + 0.5) / g).view(1, g, 1)
    w1 = (s - q).abs().mean(dim=1)  # (W, D)
    phi = torch.special.ndtr(s)
    d_plus = ((i + 1).view(1, g, 1) / g - phi).amax(dim=1)
    d_minus = (phi - i.view(1, g, 1) / g).amax(dim=1)
    ks = torch.maximum(d_plus, d_minus)  # (W, D)
    return dict(
        var_w=var.mean().item(),
        w1_w=w1.mean().item(),
        ks_w=ks.mean().item(),
        n_win=int(W * D),
    )


def global_stats(z: torch.Tensor, g: int = 32) -> dict:
    """Whole-sequence per-coordinate variance, mean-energy ('offset') and the
    share of that variance explained by window means ('drift').

    offset = mean over coordinates of (mean_t z)^2: 0 for a centred N(0,1)
    coordinate, 1 when the mean is one standard deviation away.
    drift = 1 - E[within-window variance] / E[global variance]: 0 when windows
    look like the whole sequence, -> 1 when each window is tight but windows
    sit at different levels (what per-group scaling exploits).
    """
    S, D = z.shape
    z = z.float()
    var_g = z.var(dim=0, unbiased=False)  # (D,)
    offset = (z.mean(dim=0) ** 2).mean().item()
    W = S // g
    if W >= 1:
        within = z[: W * g].reshape(W, g, D).var(dim=1, unbiased=False).mean().item()
        drift = 1.0 - within / max(var_g.mean().item(), 1e-12)
    else:
        drift = float("nan")
    return dict(var_g=var_g.mean().item(), offset=offset, drift=drift, n_tok=int(S))


def gaussian_reference(
    S: int, D: int, g: int = 32, seed: int = 0, device: str = "cpu"
) -> dict:
    """window_stats of true N(0,1) samples with the same shape (the paper's 'Gaussian' row)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    z = torch.randn(S, D, generator=gen).to(device)
    out = window_stats(z, g)
    out.update(global_stats(z, g))
    return out


# --------------------------------------------------------------------------
# quantization error of the three codebook stages
# --------------------------------------------------------------------------


def _relmse(z: torch.Tensor, zhat: torch.Tensor) -> float:
    return ((zhat - z) ** 2).sum().item() / max((z**2).sum().item(), 1e-12)


def quant_fixed(z: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """Fixed Lloyd-Max codebook (z in the sqrt(D)-scaled frame; centroids are
    for N(0, 1/D), so scale them up to match)."""
    D = z.shape[-1]
    c = centroids.to(z.device).float() * math.sqrt(D)
    idx = (z.unsqueeze(-1) - c).abs().argmin(dim=-1)
    return c[idx]


def quant_group(z: torch.Tensor, bits: int, g: int = 32) -> torch.Tensor:
    """Per-window per-coordinate min-max onto a uniform 2^bits grid (the
    paper's temporal range adaptation; same padding rule as MSECompressor)."""
    S, D = z.shape
    G = -(-S // g)
    pad = G * g - S
    zz = torch.cat([z, z[-1:].expand(pad, D)], dim=0) if pad else z
    zg = zz.reshape(G, g, D)
    gmin = zg.amin(dim=1, keepdim=True).half().float()
    gscale = (zg.amax(dim=1, keepdim=True) - gmin).clamp_min(1e-6).half().float()
    levels = (1 << bits) - 1
    idx = ((zg - gmin) / gscale * levels).round().clamp_(0, levels)
    return (idx / levels * gscale + gmin).reshape(G * g, D)[:S]


def quant_affine(
    z: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, centroids: torch.Tensor
) -> torch.Tensor:
    """Static per-coordinate affine map (calibrated mu, sigma per coordinate)
    then the fixed Lloyd-Max codebook on the standardized values."""
    sigma = sigma.clamp_min(1e-6)
    zs = (z - mu) / sigma  # ~N(0,1) if the calibration holds
    c = centroids.to(z.device).float() * math.sqrt(z.shape[-1])
    idx = (zs.unsqueeze(-1) - c).abs().argmin(dim=-1)
    return c[idx] * sigma + mu


def quant_kmeans(z: torch.Tensor, bits: int, iters: int = 20) -> torch.Tensor:
    """Per-coordinate 1-D k-means (Lloyd) with 2^bits levels fitted on z itself:
    the codebook a prefill-time quantizer could fit on the prompt (metadata =
    2^bits fp16 levels per coordinate per block)."""
    zz = z.unsqueeze(0)
    lv = fit_levels(zz, bits, "kmeans", iters)
    return reconstruct(assign(zz, lv), lv)[0]


def _error_stats(z: torch.Tensor, zhat: torch.Tensor, name: str, bits: int) -> dict:
    """relmse; shift = energy of the per-coordinate MEAN error over the key
    energy (systematic bias, 0 for an unbiased quantizer); relmsec = relmse of
    the per-coordinate-centred keys (>= 1 means the between-token differences,
    i.e. token identity inside the block, are gone)."""
    e = zhat - z
    zc = z - z.mean(dim=0, keepdim=True)
    ec = e - e.mean(dim=0, keepdim=True)
    return {
        f"relmse_{name}_b{bits}": _relmse(z, zhat),
        f"shift_{name}_b{bits}": (e.mean(dim=0) ** 2).sum().item()
        / max((z**2).mean(dim=0).sum().item(), 1e-12),
        f"relmsec_{name}_b{bits}": (ec**2).sum().item()
        / max((zc**2).sum().item(), 1e-12),
    }


def quant_errors(
    z: torch.Tensor,
    bits: int,
    centroids: torch.Tensor,
    g: int = 32,
    mu: torch.Tensor | None = None,
    sigma: torch.Tensor | None = None,
) -> dict:
    """Error stats of every codebook stage on one (S, D) coordinate block:
    fixed Lloyd-Max, per-window min-max (g), one min-max window over the whole
    block (gseg), k-means fitted on the block, static affine (if calibrated)."""
    out = {}
    out.update(_error_stats(z, quant_fixed(z, centroids), "fixed", bits))
    out.update(_error_stats(z, quant_group(z, bits, g), f"g{g}", bits))
    out.update(_error_stats(z, quant_group(z, bits, max(z.shape[0], 1)), "gseg", bits))
    out.update(_error_stats(z, quant_kmeans(z, bits), "kmeans", bits))
    if mu is not None:
        out.update(
            _error_stats(z, quant_affine(z, mu, sigma, centroids), "affine", bits)
        )
    else:
        out[f"relmse_affine_b{bits}"] = float("nan")
    return out


class AffineCalib:
    """Running per-(layer, head, coordinate) mean/std of rotated key coordinates."""

    def __init__(self):
        self.sum: dict = {}
        self.sq: dict = {}
        self.n: dict = {}

    def add(self, key: tuple, z: torch.Tensor) -> None:
        z = z.float()
        self.sum[key] = self.sum.get(key, 0) + z.sum(dim=0)
        self.sq[key] = self.sq.get(key, 0) + (z**2).sum(dim=0)
        self.n[key] = self.n.get(key, 0) + z.shape[0]

    def get(self, key: tuple):
        if key not in self.n or self.n[key] < 2:
            return None, None
        mu = self.sum[key] / self.n[key]
        var = self.sq[key] / self.n[key] - mu**2
        return mu, var.clamp_min(0).sqrt()


def codebook(head_dim: int, bits: int) -> torch.Tensor:
    return LloydMaxCodebook(head_dim, bits).centroids


# --------------------------------------------------------------------------
# alignment: query / key second moments
# --------------------------------------------------------------------------


def effective_rank(M: torch.Tensor) -> float:
    """exp(entropy of the normalized eigenvalue spectrum) of a PSD matrix."""
    ev = torch.linalg.eigvalsh(M.float()).clamp_min(0)
    p = ev / ev.sum().clamp_min(1e-12)
    return torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum()).item()


def alignment_stats(q: torch.Tensor, k: torch.Tensor, topk: int = 8) -> dict:
    """q: (Nq, D) queries, k: (Nk, D) keys of ONE kv head (raw, un-rotated).

    q_erank: effective rank of E[q q^T] (how many directions queries use).
    k_erank: effective rank of the centred key covariance.
    k_share_topq: share of centred key variance inside the top-k eigenvectors
                  of E[q q^T]; the isotropic expectation is topk / D.
    """
    q = q.float()
    k = k.float()
    D = q.shape[-1]
    Mq = q.T @ q / max(q.shape[0], 1)
    kc = k - k.mean(dim=0, keepdim=True)
    Ck = kc.T @ kc / max(k.shape[0], 1)
    _, U = torch.linalg.eigh(Mq)
    top = U[:, -topk:]
    share = torch.trace(top.T @ Ck @ top).item() / max(torch.trace(Ck).item(), 1e-12)
    return dict(
        q_erank=effective_rank(Mq),
        k_erank=effective_rank(Ck),
        k_share_topq=share,
        k_share_iso=topk / D,
    )


def attention_mass(
    query: torch.Tensor,
    key: torch.Tensor,
    groups: int,
    scaling: float,
    first: int,
    prompt_len: int,
) -> tuple[float, float]:
    """Mean (over heads) attention mass of ONE query step on the first `first`
    keys and on the prompt (keys [0, prompt_len)). query (Hq, 1, D), key (Hkv, S, D)."""
    ks = key.repeat_interleave(groups, dim=0).float()  # (Hq, S, D)
    aw = torch.softmax(
        (query.float() @ ks.transpose(1, 2)) * scaling, dim=-1
    )  # (Hq, 1, S)
    return aw[..., :first].sum(-1).mean().item(), aw[..., :prompt_len].sum(
        -1
    ).mean().item()


# --------------------------------------------------------------------------
# one sentence, one layer -> rows
# --------------------------------------------------------------------------


def layer_rows(
    layer: int,
    keys: torch.Tensor,
    values: torch.Tensor,
    keys_pre: torch.Tensor | None,
    prompt_len: int,
    Pi: torch.Tensor,
    bits_list: list,
    centroids: dict,
    g: int,
    calib: AffineCalib | None,
    evaluate_affine: bool,
    ident: dict,
) -> list:
    """keys / values / keys_pre: (Hkv, S, D) for one talker layer of one sentence.

    Returns one row per (kv_head, tensor, segment). `calib` collects affine
    statistics from the 'all' segment of post-RoPE keys when evaluate_affine is
    False (calibration sentences) and is read when True.
    """
    rows = []
    Hkv, S, D = keys.shape
    tensors = {"key": keys, "value": values}
    if keys_pre is not None:
        tensors["key_pre"] = keys_pre
    segments = {"all": (0, S), "prompt": (0, prompt_len), "gen": (prompt_len, S)}
    for h in range(Hkv):
        for name, t in tensors.items():
            z_all = rotate_unit(t[h], Pi)
            for seg, (a, b) in segments.items():
                z = z_all[a:b]
                if z.shape[0] < 2:
                    continue
                row = dict(ident, layer=layer, kv_head=h, tensor=name, segment=seg)
                row.update(window_stats(z, g))
                row.update(global_stats(z, g))
                if name.startswith("key"):
                    ck = (layer, h, name)
                    mu = sigma = None
                    if calib is not None:
                        if evaluate_affine:
                            mu, sigma = calib.get(ck)
                        elif seg == "all":
                            calib.add(ck, z)
                    for bits in bits_list:
                        row.update(quant_errors(z, bits, centroids[bits], g, mu, sigma))
                rows.append(row)
        # Gaussian reference with this head's shape (one per head, 'all')
        row = dict(ident, layer=layer, kv_head=h, tensor="gaussian", segment="all")
        row.update(
            gaussian_reference(S, D, g, seed=layer * 100 + h, device=keys.device)
        )
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# recorder (hooked into the model's eager attention) and summary
# --------------------------------------------------------------------------


class KeyStatsRecorder:
    """Collects, per talker layer, the final K/V cache, all queries, the RoPE
    tables and the per-step attention mass; `finalize` turns them into rows."""

    def __init__(
        self, talker_class, bits_list, group, first, seed, calib_sentences, rope_fn=None
    ):
        self.talker_class = talker_class
        self.bits_list, self.group, self.first, self.seed = (
            bits_list,
            group,
            first,
            seed,
        )
        self.calib_sentences = calib_sentences
        self.rope_fn = (
            rope_fn  # rope_keys(k, cos, sin, mrope_section, interleaved, inverse)
        )
        self.active = False
        self.errors, self.first_error = 0, None
        self.calib = AffineCalib()
        self.sentence_no = 0
        self.Pi, self.centroids = {}, {}
        self.seen_classes: dict = {}
        self.reset()

    def reset(self):
        self.kv: dict = {}
        self.queries: dict = {}
        self.mass: dict = {}
        self.rope = None
        self.prompt_len = None
        self.rope_scaling = None

    # -- per attention call -------------------------------------------------
    def record(self, module, query, key, value, attention_mask, scaling):
        name = type(module).__name__
        self.seen_classes[name] = self.seen_classes.get(name, 0) + 1
        if name != self.talker_class:
            return
        layer = module.layer_idx
        pos = key.shape[2]
        if query.shape[2] == pos:  # prefill
            if layer == 0:
                self.prompt_len = pos
                self.rope = module._tq_rope
                self.rope_scaling = getattr(module, "rope_scaling", None) or {}
        elif (
            layer == 0 and self.rope is not None
        ):  # first layer of a decode step: extend the tables
            cos, sin = module._tq_rope
            self.rope = (
                torch.cat([self.rope[0], cos], dim=-2),
                torch.cat([self.rope[1], sin], dim=-2),
            )
        self.kv[layer] = (
            key[0].detach(),
            value[0].detach(),
        )  # latest = full cache (Hkv, S, D)
        self.queries.setdefault(layer, []).append(query[0].detach().to(torch.float16))
        if query.shape[2] == 1 and self.prompt_len:
            m64, mp = attention_mass(
                query[0],
                key[0],
                module.num_key_value_groups,
                scaling,
                self.first,
                self.prompt_len,
            )
            s = self.mass.setdefault(layer, [0.0, 0.0, 0])
            s[0] += m64
            s[1] += mp
            s[2] += 1

    # -- per sentence ---------------------------------------------------------
    @torch.no_grad()
    def finalize(self, ident: dict) -> tuple[list, list]:
        rows, layer_stats = [], []
        evaluate_affine = self.sentence_no >= self.calib_sentences
        for layer in sorted(self.kv):
            keys, values = self.kv[layer]
            Hkv, S, D = keys.shape
            if layer not in self.Pi:
                self.Pi[layer] = layer_rotation(D, layer, self.seed, keys.device)
                for b in self.bits_list:
                    self.centroids.setdefault(b, codebook(D, b).to(keys.device))
            keys_pre = None
            if (
                self.rope_fn is not None
                and self.rope is not None
                and self.rope[0].shape[-2] == S
            ):
                cos, sin = self.rope
                rs = self.rope_scaling or {}
                keys_pre = self.rope_fn(
                    keys.unsqueeze(0),
                    cos,
                    sin,
                    mrope_section=rs.get("mrope_section") if cos.dim() == 4 else None,
                    interleaved=bool(rs.get("interleaved", False)),
                    inverse=True,
                )[0]
            rows += layer_rows(
                layer,
                keys,
                values,
                keys_pre,
                self.prompt_len or 0,
                self.Pi[layer],
                self.bits_list,
                self.centroids,
                self.group,
                self.calib,
                evaluate_affine,
                ident,
            )
            q = torch.cat(self.queries[layer], dim=1)  # (Hq, S, D)
            groups = q.shape[0] // Hkv
            ls = dict(
                ident,
                layer=layer,
                n_tok=S,
                prompt_len=self.prompt_len or 0,
                pre_rope_ok=keys_pre is not None,
            )
            al = [
                alignment_stats(
                    q[h * groups : (h + 1) * groups].reshape(-1, D), keys[h]
                )
                for h in range(Hkv)
            ]
            for k in al[0]:
                ls[k] = sum(a[k] for a in al) / Hkv
            m = self.mass.get(layer)
            ls["attn_mass_first"] = m[0] / m[2] if m and m[2] else float("nan")
            ls["attn_mass_prompt"] = m[1] / m[2] if m and m[2] else float("nan")
            layer_stats.append(ls)
        self.sentence_no += 1
        self.reset()
        return rows, layer_stats


def make_patch(original, recorder):
    def patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
        out, attn = original(
            module, query, key, value, attention_mask, scaling, dropout, **kw
        )
        if recorder.active:
            try:
                recorder.record(module, query, key, value, attention_mask, scaling)
            except Exception:  # noqa: BLE001
                recorder.errors += 1
                if recorder.first_error is None:
                    recorder.first_error = traceback.format_exc()
        return out, attn

    return patched


# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------


def _wmean(df, cols, w):
    return {c: float((df[c] * df[w]).sum() / df[w].sum()) for c in cols}


def segment_table(stats: pd.DataFrame, bits: list, g: int) -> list:
    """Prompt vs generated keys per quantizer: relmse, shift (bias energy) and
    relmsec (error on the centred keys = loss of token identity)."""
    quants = [
        ("fixed", "fixed Lloyd-Max"),
        (f"g{g}", f"min-max g{g}"),
        ("gseg", "min-max, one window per segment"),
        ("kmeans", "k-means fitted on the segment"),
        ("affine", "static affine"),
    ]
    L = [
        "## Prompt vs generated keys: does quantization shift them? (post-RoPE keys)",
        "",
        "relmse = error / energy; shift = energy of the mean error / energy (systematic bias); "
        "relmsec = error of the centred keys / their energy (1 = token identity inside the segment is gone).",
        "",
    ]
    for b in bits:
        L += [
            f"### {b}-bit keys",
            "",
            "| quantizer | prompt relmse | prompt shift | prompt relmsec | gen relmse | gen shift | gen relmsec |",
            "|---|---|---|---|---|---|---|",
        ]
        for q, label in quants:
            cols = [f"{s}_{q}_b{b}" for s in ("relmse", "shift", "relmsec")]
            if not all(c in stats.columns for c in cols):
                continue
            cells = []
            for seg in ("prompt", "gen"):
                d = stats[(stats.tensor == "key") & (stats.segment == seg)].dropna(
                    subset=cols[:1]
                )
                cells += [f"{d[c].mean():.4f}" if len(d) else "nan" for c in cols]
            L.append(f"| {label} | " + " | ".join(cells) + " |")
        L.append("")
    return L


def summarize(stats: pd.DataFrame, layers: pd.DataFrame, g: int) -> str:
    L = [
        "# Key/value Gaussianity on Qwen3-TTS (OmniKVQuant Table 1 analogue)",
        "",
        f"Windows of {g} tokens, talker layers, coordinates scaled so the codebook target is N(0, 1). "
        f"{stats['idx'].nunique()} sentences.",
        "",
        "| distribution | retained var | W1 | KS | offset | drift |",
        "|---|---|---|---|---|---|",
    ]
    order = [
        ("gaussian", "all", "Gaussian"),
        ("key", "all", "Key (post-RoPE)"),
        ("key", "prompt", "  prompt tokens"),
        ("key", "gen", "  generated tokens"),
        ("key_pre", "all", "Key (pre-RoPE)"),
        ("value", "all", "Value"),
    ]
    for tensor, seg, label in order:
        d = stats[(stats.tensor == tensor) & (stats.segment == seg) & (stats.n_win > 0)]
        if not len(d):
            continue
        w = _wmean(d, ["var_w", "w1_w", "ks_w"], "n_win")
        gl = _wmean(d, ["offset", "drift"], "n_tok")
        L.append(
            f"| {label} | {100 * w['var_w']:.2f}% | {w['w1_w']:.3f} | {w['ks_w']:.3f} | {gl['offset']:.3f} | {gl['drift']:.3f} |"
        )
    L += [
        "",
        "Retained var of a true N(0,1) window is (g-1)/g; W1/KS of the Gaussian row are the finite-sample floor.",
        "offset = energy of the per-coordinate mean (0 = centred); drift = share of variance between windows.",
        "",
    ]

    bits = sorted(
        {
            int(c.split("_b")[-1])
            for c in stats.columns
            if c.startswith("relmse_fixed_b")
        }
    )
    d = stats[(stats.tensor == "key") & (stats.segment == "all")]
    if len(d) and bits:
        L += [
            "## Key quantization error by codebook stage (relative MSE, post-RoPE keys)",
            "",
            "| bits | fixed Lloyd-Max | per-window min-max (g32) | static affine + Lloyd-Max | n sentences (affine) |",
            "|---|---|---|---|---|",
        ]
        for b in bits:
            da = d.dropna(subset=[f"relmse_affine_b{b}"])
            L.append(
                f"| {b} | {d[f'relmse_fixed_b{b}'].mean():.4f} | {d[f'relmse_g{g}_b{b}'].mean():.4f} | "
                f"{da[f'relmse_affine_b{b}'].mean() if len(da) else float('nan'):.4f} | {da['idx'].nunique()} |"
            )
        L += [
            "",
            "Affine = per-coordinate mean/std calibrated on the first sentences, evaluated on the rest (zero per-token metadata).",
            "",
        ]
        L += segment_table(stats, bits, g)
        L += [
            f"## Per layer, {bits[0]}-bit keys",
            "",
            "| layer | retained var | offset | drift | fixed | g32 | affine | q erank | k erank | k share in top-8 q dirs (iso) | attn mass first-64 | attn mass prompt |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        b = bits[0]
        for layer, dl in d.groupby("layer"):
            w = _wmean(dl[dl.n_win > 0], ["var_w"], "n_win")
            gl = _wmean(dl, ["offset", "drift"], "n_tok")
            ll = layers[layers.layer == layer]
            L.append(
                f"| {int(layer)} | {100 * w['var_w']:.1f}% | {gl['offset']:.2f} | {gl['drift']:.2f} | "
                f"{dl[f'relmse_fixed_b{b}'].mean():.4f} | {dl[f'relmse_g{g}_b{b}'].mean():.4f} | {dl[f'relmse_affine_b{b}'].mean():.4f} | "
                f"{ll.q_erank.mean():.1f} | {ll.k_erank.mean():.1f} | {100 * ll.k_share_topq.mean():.1f}% ({100 * ll.k_share_iso.mean():.1f}%) | "
                f"{ll.attn_mass_first.mean():.3f} | {ll.attn_mass_prompt.mean():.3f} |"
            )
    return "\n".join(L)
