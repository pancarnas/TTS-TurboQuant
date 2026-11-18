"""Levels fitted on a block of tokens: per-coordinate 1-D k-means or one min-max window.

Shapes: z (N, S, D) float32; levels (N, L, D) with L = 2**bits, sorted ascending.
"""

from __future__ import annotations

import torch


def minmax_levels(lo: torch.Tensor, hi: torch.Tensor, bits: int) -> torch.Tensor:
    """(N, L, D) evenly spaced from lo (N, 1, D) to hi (N, 1, D)."""
    num = 1 << bits
    t = torch.linspace(0, 1, num, device=lo.device, dtype=lo.dtype).view(1, num, 1)
    return lo + (hi - lo) * t


def assign(z: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """(N, S, D) int32 index of the nearest level per element."""
    lv = levels.transpose(1, 2).unsqueeze(1)  # (N, 1, D, L)
    return (z.unsqueeze(-1) - lv).abs().argmin(-1).to(torch.int32)


def reconstruct(indices: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """(N, S, D) float32 level values for the given indices."""
    return torch.gather(levels, 1, indices.long())


def meta_scalars(bits: int, mode: str) -> int:
    """Per-coordinate metadata scalars stored for one block."""
    return 1 << bits if mode == "kmeans" else 2


def _kmeans_levels(z: torch.Tensor, bits: int, iters: int) -> torch.Tensor:
    num = 1 << bits
    qs = (torch.arange(num, device=z.device, dtype=z.dtype) + 0.5) / num
    levels = torch.quantile(z, qs, dim=1).permute(1, 0, 2).contiguous()  # (N, L, D)
    for _ in range(iters):
        onehot = torch.nn.functional.one_hot(assign(z, levels).long(), num).to(z.dtype)
        cnt = onehot.sum(1)  # (N, D, L)
        mean = torch.einsum("nsdl,nsd->ndl", onehot, z) / cnt.clamp_min(1)
        levels = torch.where(cnt > 0, mean, levels.transpose(1, 2)).transpose(1, 2)
    return levels.sort(dim=1).values


def fit_levels(z: torch.Tensor, bits: int, mode: str, iters: int = 20) -> torch.Tensor:
    """Fit (N, L, D) float32 levels on z (N, S, D) with mode 'kmeans' or 'minmax'."""
    z = z.float()
    if mode == "minmax":
        return minmax_levels(z.amin(1, keepdim=True), z.amax(1, keepdim=True), bits)
    if mode == "kmeans":
        return _kmeans_levels(z, bits, iters)
    raise ValueError(f"unknown mode {mode!r}")
