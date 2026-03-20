"""Datasets for Heck et al. baseline encoder Phase-1 (triplet) training."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

from Encoder.baseline.model import get_edge_strip, load_piece


@dataclass
class EncoderTrainConfig:
    """Hyperparameters for encoder triplet training."""

    dataset_path: str
    checkpoint_dir: str
    n_pieces: int
    grid_rows: int
    grid_cols: int
    piece_size: int = 128
    strip_w: int = 8
    embed_dim: int = 320
    d: int = 1024
    margin_alpha: float = 0.5
    lr_phase1: float = 1e-4
    batch_size_triplets: int = 128
    epochs_phase1: int = 20
    train_ratio: float = 0.8
    num_workers: int = 0
    seed: int = 42
    checkpoint_name: str = "encoder_phase1.pt"


def infer_grid(n_pieces: int) -> Tuple[int, int]:
    """Infer a rectangular grid whose product equals ``n_pieces`` (same rule as the notebook)."""
    rows = max(1, int(math.sqrt(n_pieces)))
    cols = math.ceil(n_pieces / rows)
    if rows * cols != n_pieces:
        raise ValueError(
            f"Cannot infer grid for n_pieces={n_pieces}: "
            f"rows={rows}, cols={cols} gives product {rows * cols}"
        )
    return rows, cols


class PuzzleDataset(Dataset):
    """
    Loads puzzle directories with ``pieces/piece_000.png`` … ``piece_{N-1:03d}.png``.

    ``__getitem__`` returns shuffled pieces and permutation targets (for Phase 2 in the notebook).
    Phase 1 only uses ``load_ordered_pieces`` via TripletDataset.
    """

    def __init__(self, puzzle_dirs: List[Path], cfg: EncoderTrainConfig, augment: bool = False):
        self.puzzle_dirs = puzzle_dirs
        self.cfg = cfg
        self.color_jitter = (
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
            if augment
            else None
        )

    def load_ordered_pieces(self, pieces_dir: Path) -> torch.Tensor:
        """Load all N pieces in ground-truth order → [N, 3, S, S]."""
        pieces = []
        for i in range(self.cfg.n_pieces):
            t = load_piece(pieces_dir / f"piece_{i:03d}.png", self.cfg.piece_size)
            if self.color_jitter is not None and random.random() > 0.5:
                t = self.color_jitter(t)
            pieces.append(t)
        return torch.stack(pieces)

    def __len__(self) -> int:
        return len(self.puzzle_dirs)

    def __getitem__(self, idx: int):
        pieces_dir = self.puzzle_dirs[idx] / "pieces"
        ordered = self.load_ordered_pieces(pieces_dir)
        n = self.cfg.n_pieces
        perm = torch.randperm(n)
        shuffled = ordered[perm]
        p_target = torch.zeros(n, n)
        p_target[perm, torch.arange(n)] = 1.0
        return shuffled, p_target, ordered


class TripletDataset(Dataset):
    """
    (anchor, positive, negative) edge-strip triplets for encoder pre-training.
    """

    def __init__(self, base_ds: PuzzleDataset, cfg: EncoderTrainConfig):
        self.base = base_ds
        self.cfg = cfg
        rows, cols = cfg.grid_rows, cfg.grid_cols
        self.specs: List[Tuple[int, int, int, str, str]] = []
        for pi in range(len(base_ds)):
            for r in range(rows):
                for c in range(cols):
                    a = r * cols + c
                    if c + 1 < cols:
                        self.specs.append((pi, a, r * cols + (c + 1), "right", "left"))
                    if r + 1 < rows:
                        self.specs.append((pi, a, (r + 1) * cols + c, "bottom", "top"))

    def __len__(self) -> int:
        return len(self.specs)

    def __getitem__(self, idx: int):
        pi, a_idx, b_idx, anchor_dir, pos_dir = self.specs[idx]
        pieces = self.base.load_ordered_pieces(self.base.puzzle_dirs[pi] / "pieces")
        n = self.cfg.n_pieces
        sw = self.cfg.strip_w
        anchor = get_edge_strip(pieces[a_idx], anchor_dir, sw)
        positive = get_edge_strip(pieces[b_idx], pos_dir, sw)
        neg_idx = b_idx
        while neg_idx == b_idx:
            neg_idx = random.randint(0, n - 1)
        negative = get_edge_strip(pieces[neg_idx], pos_dir, sw)
        return anchor, positive, negative


def make_datasets(cfg: EncoderTrainConfig) -> Tuple[PuzzleDataset, PuzzleDataset]:
    """Scan ``cfg.dataset_path`` and return ``(train_ds, test_ds)``."""
    base = Path(cfg.dataset_path)
    if not base.exists():
        raise FileNotFoundError(f"Dataset path not found: {base}")

    puzzle_dirs = sorted(
        [
            d
            for d in base.iterdir()
            if d.is_dir()
            and (d / "pieces").is_dir()
            and len(list((d / "pieces").glob("piece_*.png"))) == cfg.n_pieces
        ]
    )
    if not puzzle_dirs:
        raise RuntimeError(
            f"No complete puzzles found in {base}.\n"
            f"Expected: <name>/pieces/piece_000.png … piece_{cfg.n_pieces - 1:03d}.png"
        )

    rng = random.Random(cfg.seed)
    rng.shuffle(puzzle_dirs)
    n_train = int(len(puzzle_dirs) * cfg.train_ratio)
    train_ds = PuzzleDataset(puzzle_dirs[:n_train], cfg, augment=True)
    test_ds = PuzzleDataset(puzzle_dirs[n_train:], cfg, augment=False)
    print(f"Found {len(puzzle_dirs)} puzzles → train: {n_train}, test: {len(test_ds)}")
    return train_ds, test_ds


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
