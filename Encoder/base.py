"""Abstract base for puzzle-piece encoders used with PuzzleNet."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import numpy as np


class BaseEncoder(ABC):
    """Per-piece embedding dimension (after concatenation if multiple branches)."""

    embed_dim: int

    @abstractmethod
    def encode_puzzle(self, piece_paths: List[Path]) -> np.ndarray:
        """
        Encode all pieces of one puzzle in sorted path order.

        Returns
        -------
        np.ndarray of shape (N_pieces, embed_dim), float32.
        """
