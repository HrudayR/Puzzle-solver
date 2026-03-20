"""Losses for Heck et al. baseline encoder training (Phase 1 triplet)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def cosine_dissimilarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Element-wise  1 − cos_sim(a, b).  Shape: same as inputs minus last dim."""
    return 1.0 - F.cosine_similarity(a, b, dim=-1)


def triplet_loss(
    za: torch.Tensor,
    zp: torch.Tensor,
    zn: torch.Tensor,
    margin: float = 0.5,
) -> torch.Tensor:
    """Triplet loss with cosine dissimilarity (Eq. 4 from paper)."""
    return F.relu(cosine_dissimilarity(za, zp) - cosine_dissimilarity(za, zn) + margin).mean()
