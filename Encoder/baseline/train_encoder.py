#!/usr/bin/env python3
"""
Phase-1 training for the Heck et al. CNN edge encoder (triplet loss on edge strips).

Saves ``encoder_phase1.pt`` under ``--checkpoint-dir``.

Example:
  python -m Encoder.baseline.train_encoder \\
    --dataset-path ./Dataset/train_set_square \\
    --n-pieces 20 \\
    --checkpoint-dir ./Encoder/baseline/checkpoints
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Project root on path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Encoder.baseline.data import (
    EncoderTrainConfig,
    TripletDataset,
    infer_grid,
    make_datasets,
    set_seed,
)
from Encoder.baseline.losses import cosine_dissimilarity, triplet_loss
from Encoder.baseline.model import PieceEncoder


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_phase1(cfg: EncoderTrainConfig, device: torch.device) -> PieceEncoder:
    train_ds, _test_ds = make_datasets(cfg)
    triplet_ds = TripletDataset(train_ds, cfg)
    loader = DataLoader(
        triplet_ds,
        batch_size=cfg.batch_size_triplets,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    encoder = PieceEncoder(cfg.embed_dim, cfg.d, cfg.strip_w).to(device)
    optimizer = optim.Adam(encoder.parameters(), lr=cfg.lr_phase1)
    use_amp_cuda = torch.cuda.is_available()
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp_cuda)
    except (TypeError, AttributeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp_cuda)

    out_dir = Path(cfg.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "encoder_phase1.pt"
    history: dict = {"loss": [], "top1_acc": []}

    print(f"\n{'='*60}")
    print("Phase 1 – encoder pre-training (triplet loss)")
    print(f"  Triplets : {len(triplet_ds):,}")
    print(f"  Epochs   : {cfg.epochs_phase1}")
    print(f"  Batch    : {cfg.batch_size_triplets}")
    print(f"  Device   : {device}")
    print(f"{'='*60}\n")

    for epoch in range(1, cfg.epochs_phase1 + 1):
        encoder.train()
        run_loss = 0.0
        correct = 0
        total = 0
        t0 = time.time()

        for anchor, positive, negative in tqdm(
            loader, desc=f"Epoch {epoch}/{cfg.epochs_phase1}", leave=False
        ):
            anchor = anchor.to(device, non_blocking=True)
            positive = positive.to(device, non_blocking=True)
            negative = negative.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if use_amp_cuda:
                with torch.amp.autocast("cuda", enabled=True):
                    za = encoder.cnn(anchor)
                    zp = encoder.cnn(positive)
                    zn = encoder.cnn(negative)
                    loss = triplet_loss(za, zp, zn, margin=cfg.margin_alpha)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                za = encoder.cnn(anchor)
                zp = encoder.cnn(positive)
                zn = encoder.cnn(negative)
                loss = triplet_loss(za, zp, zn, margin=cfg.margin_alpha)
                loss.backward()
                optimizer.step()

            run_loss += loss.item()
            with torch.no_grad():
                d_pos = cosine_dissimilarity(za, zp)
                d_neg = cosine_dissimilarity(za, zn)
                correct += (d_pos < d_neg).sum().item()
                total += anchor.size(0)

        avg_loss = run_loss / len(loader)
        top1_acc = correct / total * 100 if total else 0.0
        history["loss"].append(avg_loss)
        history["top1_acc"].append(top1_acc)
        print(
            f"  Epoch {epoch:2d}/{cfg.epochs_phase1} │ "
            f"loss={avg_loss:.4f}  top-1={top1_acc:.1f}%  ({time.time() - t0:.0f}s)"
        )

    payload = {
        "encoder_state": encoder.state_dict(),
        "history": history,
        "cfg": {
            "dataset_path": cfg.dataset_path,
            "n_pieces": cfg.n_pieces,
            "grid_rows": cfg.grid_rows,
            "grid_cols": cfg.grid_cols,
            "piece_size": cfg.piece_size,
            "strip_w": cfg.strip_w,
            "embed_dim": cfg.embed_dim,
            "d": cfg.d,
        },
    }
    try:
        torch.save(payload, ckpt_path, weights_only=False)
    except TypeError:
        torch.save(payload, ckpt_path)
    print(f"\nCheckpoint saved → {ckpt_path}")
    return encoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Heck et al. PieceEncoder (Phase 1 triplet).")
    parser.add_argument("--dataset-path", type=str, required=True, help="Root with puzzle subfolders.")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(_ROOT / "Encoder/baseline/checkpoints"),
        help="Directory for encoder_phase1.pt",
    )
    parser.add_argument("--n-pieces", type=int, default=20, help="Pieces per puzzle (must match files).")
    parser.add_argument("--grid-rows", type=int, default=None, help="Override inferred grid rows.")
    parser.add_argument("--grid-cols", type=int, default=None, help="Override inferred grid cols.")
    parser.add_argument("--piece-size", type=int, default=128)
    parser.add_argument("--strip-w", type=int, default=8)
    parser.add_argument("--embed-dim", type=int, default=320)
    parser.add_argument("--d", type=int, default=1024, dest="d_model", help="Piece embedding dimension.")
    parser.add_argument("--margin", type=float, default=0.5, help="Triplet margin (MARGIN_ALPHA).")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.grid_rows is not None and args.grid_cols is not None:
        grid_rows, grid_cols = args.grid_rows, args.grid_cols
    else:
        grid_rows, grid_cols = infer_grid(args.n_pieces)

    cfg = EncoderTrainConfig(
        dataset_path=args.dataset_path,
        checkpoint_dir=args.checkpoint_dir,
        n_pieces=args.n_pieces,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        piece_size=args.piece_size,
        strip_w=args.strip_w,
        embed_dim=args.embed_dim,
        d=args.d_model,
        margin_alpha=args.margin,
        lr_phase1=args.lr,
        batch_size_triplets=args.batch_size,
        epochs_phase1=args.epochs,
        train_ratio=args.train_ratio,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    set_seed(cfg.seed)
    device = pick_device()
    train_phase1(cfg, device)


if __name__ == "__main__":
    main()
