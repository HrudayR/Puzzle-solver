"""
Heck et al. 2025 — CNN edge encoder (PieceEncoder + EdgeCNN).

Heck et al. CNN edge encoder — used for inference and Phase-1 training in this package.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image


def rgba_to_rgb(img: Image.Image) -> Image.Image:
    """Composite an RGBA image onto a white background → RGB."""
    if img.mode == "RGB":
        return img
    if img.mode != "RGBA":
        return img.convert("RGB")
    bg = Image.new("RGB", img.size, (255, 255, 255))
    bg.paste(img, mask=img.split()[3])
    return bg


def load_piece(path: Path, size: int) -> torch.Tensor:
    """Load one puzzle piece PNG → float32 [3, size, size] tensor in [0, 1]."""
    img = rgba_to_rgb(Image.open(path))
    img = img.resize((size, size), Image.LANCZOS)
    return T.ToTensor()(img)  # [3, size, size]


def get_edge_strip(piece: torch.Tensor, direction: str, strip_w: int) -> torch.Tensor:
    """
    Extract an edge strip from piece [3, H, W], applying a geometric transform
    so the edge of interest always ends up on the RIGHT side of the output strip.
    """
    if direction == "right":
        return piece[..., -strip_w:]
    if direction == "left":
        return piece.flip(-1)[..., -strip_w:]
    if direction == "bottom":
        return piece.rot90(1, dims=[-2, -1])[..., -strip_w:]
    if direction == "top":
        return piece.flip(-2).rot90(1, dims=[-2, -1])[..., -strip_w:]
    raise ValueError(f"Unknown direction: {direction!r}")


class EdgeCNN(nn.Module):
    """
    Encodes a single edge strip [B, 3, H, strip_w] → [B, embed_dim].

    Architecture (paper):
        Conv(3×3)+ReLU → MaxPool(2×2)   [×2]
        Conv(3×3)+ReLU
        AdaptiveAvgPool(2×2)
        FC → embed_dim
    """

    def __init__(self, embed_dim: int = 320):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.fc = nn.Linear(256 * 2 * 2, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convs(x)
        x = self.pool(x)
        return self.fc(x.flatten(1))


class PieceEncoder(nn.Module):
    """
    Encodes a puzzle piece [3, S, S] into a d-dimensional vector by:
        1. Extracting edge strips for all 4 directions (right/left/bottom/top).
        2. Encoding each strip with the shared EdgeCNN  → 4 × embed_dim vectors.
        3. Concatenating and projecting through an FC layer → d-dim vector.
    """

    def __init__(self, embed_dim: int = 320, d: int = 1024, strip_w: int = 8):
        super().__init__()
        self.strip_w = strip_w
        self.cnn = EdgeCNN(embed_dim)
        self.proj = nn.Linear(4 * embed_dim, d)

    @staticmethod
    def _extract_strips(
        pieces: torch.Tensor, strip_w: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (right, left, bottom, top) strips for [B, 3, S, S] batch."""
        r = pieces[..., -strip_w:]
        l = pieces.flip(-1)[..., -strip_w:]
        b = pieces.rot90(1, dims=[-2, -1])[..., -strip_w:]
        t = pieces.flip(-2).rot90(1, dims=[-2, -1])[..., -strip_w:]
        return r, l, b, t

    def encode_strips(self, pieces: torch.Tensor) -> torch.Tensor:
        """pieces: [B, 3, S, S]  →  [B, d]"""
        r, l, b, t = self._extract_strips(pieces, self.strip_w)
        z = torch.cat([self.cnn(r), self.cnn(l), self.cnn(b), self.cnn(t)], dim=-1)
        return self.proj(z)

    def forward(self, pieces: torch.Tensor) -> torch.Tensor:
        """pieces: [B, N, 3, S, S]  →  [B, N, d]"""
        b, n, c, s, _ = pieces.shape
        z = self.encode_strips(pieces.reshape(b * n, c, s, s))
        return z.reshape(b, n, -1)
