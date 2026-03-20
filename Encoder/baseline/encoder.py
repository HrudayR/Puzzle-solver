"""Heck et al. PieceEncoder with frozen weights from a Phase-1 checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch

from Encoder.base import BaseEncoder

from .model import PieceEncoder, load_piece

_DEFAULT_PIECE_SIZE = 128
_DEFAULT_STRIP_W = 8
_DEFAULT_EMBED_DIM = 320
_DEFAULT_D = 1024


class BaselineEncoder(BaseEncoder):
    """Frozen PieceEncoder; embed_dim matches checkpoint projection output (default 1024)."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | torch.device | None = None,
        piece_size: int = _DEFAULT_PIECE_SIZE,
        strip_w: int = _DEFAULT_STRIP_W,
        embed_dim: int = _DEFAULT_EMBED_DIM,
        d: int = _DEFAULT_D,
    ) -> None:
        if device is not None:
            self._device = torch.device(device)
        elif torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")
        self._piece_size = piece_size
        self.encoder = PieceEncoder(embed_dim=embed_dim, d=d, strip_w=strip_w).to(self._device)
        try:
            ckpt = torch.load(checkpoint_path, map_location=self._device, weights_only=False)
        except TypeError:
            ckpt = torch.load(checkpoint_path, map_location=self._device)
        state = ckpt["encoder_state"] if isinstance(ckpt, dict) and "encoder_state" in ckpt else ckpt
        self.encoder.load_state_dict(state, strict=True)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.embed_dim = d

    def encode_puzzle(self, piece_paths: List[Path]) -> np.ndarray:
        tensors = torch.stack(
            [load_piece(Path(p), self._piece_size) for p in piece_paths],
            dim=0,
        ).to(self._device)
        with torch.no_grad():
            z = self.encoder(tensors.unsqueeze(0)).squeeze(0)
        return z.cpu().numpy().astype(np.float32)
