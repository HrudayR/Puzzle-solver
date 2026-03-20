#!/usr/bin/env python3
"""
Train PuzzleNet on shuffled piece embeddings with a frozen encoder (ours or baseline).

Example:
  python -m Network.train --encoder ours --dataset-path ./Dataset/train_set_curved --num-pieces 20
  python Network/train.py --encoder baseline_square --checkpoint Encoder/baseline/checkpoints/encoder_phase1.pt
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.optim as optim

# Project root (parent of Network/)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Encoder.base import BaseEncoder
from Encoder.baseline import BaselineEncoder
from Network.puzzle_net import PuzzleNet, sinkhorn, sinkhorn_loss


def find_puzzle_dirs(root: Path, num_pieces: int) -> List[Path]:
    """Puzzle directories under root containing exactly `num_pieces` piece PNGs."""

    out: List[Path] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        pieces_dir = d / "pieces"
        if not pieces_dir.is_dir():
            continue
        n = len(list(pieces_dir.glob("piece_*.png")))
        if n == num_pieces:
            out.append(d)
    return out


def piece_paths_for_puzzle(puzzle_dir: Path, num_pieces: int) -> List[Path]:
    paths = [puzzle_dir / "pieces" / f"piece_{i:03d}.png" for i in range(num_pieces)]
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(f"Missing piece file: {p}")
    return paths


def shuffle_puzzle(
    rng: np.random.RandomState,
    embs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    embs: (num_pieces, embed_dim) in canonical order.
    Returns shuffled flat vector (num_pieces * embed_dim,) and permutation target (N, N) one-hot.
    """
    num_pieces, _embed_dim = embs.shape
    perm = rng.permutation(num_pieces)
    shuffled = embs[perm].reshape(-1).astype(np.float32)
    target = np.zeros((num_pieces, num_pieces), dtype=np.float32)
    target[np.arange(num_pieces), perm] = 1.0
    return shuffled, target


def build_encoder(args: argparse.Namespace, device: torch.device) -> BaseEncoder:
    if args.encoder == "ours":
        from Encoder.ours import OurEncoder

        return OurEncoder(device=device)
    if args.encoder == "baseline_square":
        ckpt = Path(args.checkpoint or _ROOT / "Encoder/baseline/checkpoints/encoder_phase1.pt")
        return BaselineEncoder(ckpt, device=device)
    if args.encoder == "baseline_curved":
        ckpt = Path(
            args.checkpoint
            or _ROOT / "Encoder/baseline/checkpoints/encoder_phase1_curved.pt"
        )
        if not ckpt.is_file():
            raise FileNotFoundError(
                f"Baseline curved checkpoint not found: {ckpt}\n"
                "Train the encoder separately in the notebook, or pass --checkpoint path."
            )
        return BaselineEncoder(ckpt, device=device)
    raise ValueError(f"Unknown encoder: {args.encoder}")


def permutation_accuracy(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> Tuple[float, float]:
    """
    logits, target: (B, N, N)
    Returns mean assignment accuracy and fraction of puzzles with all positions correct.
    """
    pred = sinkhorn(logits).argmax(dim=-1)
    true_j = target.argmax(dim=-1)
    match = (pred == true_j).float()
    acc = match.mean().item()
    solved = match.all(dim=-1).float().mean().item()
    return acc, solved


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PuzzleNet with a frozen encoder.")
    parser.add_argument(
        "--encoder",
        choices=["ours", "baseline_square", "baseline_curved"],
        default="ours",
        help="Which frozen encoder to use for embeddings.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=_ROOT / "Dataset/train_set_curved",
        help="Root directory containing puzzle subfolders with pieces/.",
    )
    parser.add_argument("--num-pieces", type=int, default=20)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to baseline PieceEncoder checkpoint (required for baseline_curved if default is missing).",
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, help="cuda | cpu | mps (default: auto)")
    parser.add_argument(
        "--max-puzzles",
        type=int,
        default=None,
        help="Optional cap on number of puzzles (for faster dev runs).",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    encoder = build_encoder(args, device)
    puzzle_dirs = find_puzzle_dirs(args.dataset_path, args.num_pieces)
    if not puzzle_dirs:
        raise RuntimeError(
            f"No puzzles with {args.num_pieces} pieces under {args.dataset_path}"
        )
    if args.max_puzzles is not None:
        puzzle_dirs = puzzle_dirs[: args.max_puzzles]

    print(f"Found {len(puzzle_dirs)} puzzles → encoder={args.encoder}, embed_dim={encoder.embed_dim}")

    # Build dataset (one shuffle per puzzle)
    X_list: List[np.ndarray] = []
    Y_list: List[np.ndarray] = []
    rng = np.random.RandomState(args.seed)
    for i, pdir in enumerate(puzzle_dirs):
        paths = piece_paths_for_puzzle(pdir, args.num_pieces)
        embs = encoder.encode_puzzle(paths)
        if embs.shape != (args.num_pieces, encoder.embed_dim):
            raise ValueError(f"Bad embedding shape for {pdir}: {embs.shape}")
        shuffled, target = shuffle_puzzle(np.random.RandomState(args.seed + i), embs)
        X_list.append(shuffled)
        Y_list.append(target)

    X = np.stack(X_list, axis=0)
    Y = np.stack(Y_list, axis=0)
    n = X.shape[0]
    order = rng.permutation(n)
    X, Y = X[order], Y[order]

    split = int(args.train_ratio * n)
    if split == 0 or split == n:
        raise ValueError("Train/test split produced an empty split; need more puzzles.")
    x_train = torch.tensor(X[:split], dtype=torch.float32)
    y_train = torch.tensor(Y[:split], dtype=torch.float32)
    x_test = torch.tensor(X[split:], dtype=torch.float32)
    y_test = torch.tensor(Y[split:], dtype=torch.float32)

    print(f"Train: {x_train.shape[0]} | Test: {x_test.shape[0]} | X dim: {x_train.shape[1]}")

    d_in = args.num_pieces * encoder.embed_dim
    model = PuzzleNet(d=d_in, pair_dim=1, num_pieces=args.num_pieces).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    x_train_d = x_train.to(device)
    y_train_d = y_train.to(device)
    x_test_d = x_test.to(device)
    y_test_d = y_test.to(device)

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(x_train_d)
        loss = sinkhorn_loss(logits, y_train_d)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            t_logits = model(x_test_d)
            t_loss = sinkhorn_loss(t_logits, y_test_d)
            acc, solved = permutation_accuracy(t_logits, y_test_d)

        if (epoch + 1) % max(1, args.epochs // 10) == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1}/{args.epochs} | "
                f"train_loss={loss.item():.4f} | test_loss={t_loss.item():.4f} | "
                f"test_acc={acc:.4f} | test_solved={solved:.4f}"
            )

    model.eval()
    with torch.no_grad():
        t_logits = model(x_test_d)
        t_loss = sinkhorn_loss(t_logits, y_test_d)
        acc, solved = permutation_accuracy(t_logits, y_test_d)
    print(
        f"\nFinal test | loss={t_loss.item():.4f} | "
        f"per-row_acc={acc:.4f} | frac_perfect={solved:.4f}"
    )


if __name__ == "__main__":
    main()
