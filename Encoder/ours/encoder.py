"""Gabor (conv) + Fourier descriptor embeddings concatenated per piece (128-dim)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from Embeddings.conv_embeddings import GaborEncoder
from Embeddings.fourier_descriptor import embed_image

from Encoder.base import BaseEncoder

_GABOR_DIM = 64
_FOURIER_DIM = 64


_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)


class OurEncoder(BaseEncoder):
    """
    Fixed Gabor encoder + Fourier magnitudes per piece, concatenated → 128-dim.
    No learnable parameters.
    """

    embed_dim = _GABOR_DIM + _FOURIER_DIM

    def __init__(self, device: str | torch.device | None = None) -> None:
        if device is not None:
            self._device = torch.device(device)
        elif torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")
        self._gabor = GaborEncoder(embed_dim=_GABOR_DIM).to(self._device).eval()

    def _gabor_vec(self, path: Path) -> np.ndarray:
        t = _transform(Image.open(path).convert("RGB")).unsqueeze(0).to(self._device)
        with torch.no_grad():
            v = self._gabor(t).squeeze(0).cpu().numpy()
        return v.astype(np.float32)

    def encode_puzzle(self, piece_paths: List[Path]) -> np.ndarray:
        rows: List[np.ndarray] = []
        for p in piece_paths:
            g = self._gabor_vec(p)
            f = embed_image(p, embedding_size=_FOURIER_DIM)
            if f is None:
                raise RuntimeError(f"Fourier embedding failed for {p}")
            rows.append(np.concatenate([g, f.astype(np.float32)], axis=0))
        return np.stack(rows, axis=0)
