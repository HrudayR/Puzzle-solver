"""
Convolutional Embeddings Extractor (PyTorch) — Learnable CNN
=============================================================
Extracts shape + texture embeddings from puzzle piece images using a small
learnable CNN trained with a self-supervised contrastive objective.

The encoder is trained to produce similar embeddings for augmented views of
the same piece and dissimilar embeddings for different pieces (NT-Xent loss).
Once trained, weights are frozen and the encoder is used purely for inference.

Configure the script by editing the GLOBAL CONFIGURATION section below.
"""

# ===========================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
CHECKPOINT_PATH = ENCODER_CHECKPOINT if ENCODER_CHECKPOINT is not None else str(OUTPUT_DIR / "piece_encoder.pt")

import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image


# ─────────────────────────────────────────────────────────────
#  ENCODER ARCHITECTURE
#  Designed for puzzle pieces:
#    - Alpha channel (transparency mask) is used as a 4th input channel
#      so the network can explicitly see the silhouette / connector tabs
#    - Small receptive field early layers capture local edge & texture cues
#    - Deeper layers aggregate connector shape globally
# ─────────────────────────────────────────────────────────────

class PieceEncoder(nn.Module):
    """
    Lightweight CNN that embeds a puzzle piece image into a unit-norm vector.

    Input : (B, 4, H, W)  — RGBA, values in [-1, 1]
    Output: (B, embed_dim) — L2-normalised

    Learnable parameters: ~180 K (fast to train, small memory footprint)
    """

    def __init__(self, embed_dim: int = 32):
        super().__init__()

        # 4-channel input (RGB + alpha mask) so the net sees silhouette
        self.backbone = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embedder = nn.Linear(64, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.pool(x).flatten(1)
        return F.normalize(self.embedder(x), p=2, dim=1)

    def forward_proj(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for forward — kept for compatibility with train_encoder."""
        return self.forward(x)


# ─────────────────────────────────────────────────────────────
#  IMAGE LOADING — RGBA
#  We keep the alpha channel so the encoder sees the silhouette
#  directly rather than having to infer it from the background.
# ─────────────────────────────────────────────────────────────

_to_tensor = transforms.ToTensor()  # HWC uint8 → CHW float32 in [0,1]

def _normalise(tensor: torch.Tensor) -> torch.Tensor:
    """Normalise all 4 channels to [-1, 1]."""
    return tensor * 2.0 - 1.0

def load_rgba(path: Path) -> torch.Tensor:
    """Load image as (4, H, W) float32 tensor in [-1, 1]."""
    img = Image.open(path).convert("RGBA")
    return _normalise(_to_tensor(img))


# ─────────────────────────────────────────────────────────────
#  AUGMENTATION — piece-aware
#  Augmentations must preserve the overall silhouette so the
#  encoder learns meaningful shape features rather than learning
#  to ignore all spatial information.
# ─────────────────────────────────────────────────────────────

class PieceAugmentation:
    """
    Produces two differently-augmented views of the same piece image
    for self-supervised contrastive training.

    What we do:
      - Random horizontal/vertical flips   (plausible puzzle orientations)
      - Mild random rotation ±15°          (slight orientation variation)
      - Random crop + resize               (zoom / translation invariance)
      - Color jitter on RGB only           (lighting variation)
      - Alpha channel is carried through unchanged in all spatial ops
        so the silhouette is preserved

    What we deliberately avoid:
      - Strong rotation (>30°): connector tabs are orientation-sensitive
      - Cutout / erasing: would destroy the silhouette signal
      - Strong blur: the tab edges are important fine-detail
    """

    def __init__(self, size: int = 128):
        self.size = size

        self.color_jitter = transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.05,
        )

    def __call__(self, img_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            img_tensor: (4, H, W) float32 in [-1, 1]
        Returns:
            (view1, view2): two augmented tensors of the same shape
        """
        return self._augment(img_tensor), self._augment(img_tensor)

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        rgb, alpha = x[:3], x[3:4]   # split channels

        # --- spatial transforms (applied to all channels together) ---
        # Flip
        if torch.rand(1) > 0.5:
            rgb   = TF.hflip(rgb)
            alpha = TF.hflip(alpha)
        if torch.rand(1) > 0.5:
            rgb   = TF.vflip(rgb)
            alpha = TF.vflip(alpha)

        # Mild rotation (±15°)
        angle = (torch.rand(1).item() * 30) - 15
        rgb   = TF.rotate(rgb,   angle, fill=[-1, -1, -1])
        alpha = TF.rotate(alpha, angle, fill=[-1.0])

        # Random resized crop
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            rgb,
            scale=(0.75, 1.0),
            ratio=(0.9, 1.1),
        )
        rgb   = TF.resized_crop(rgb,   i, j, h, w, [self.size, self.size],
                                interpolation=TF.InterpolationMode.BILINEAR)
        alpha = TF.resized_crop(alpha, i, j, h, w, [self.size, self.size],
                                interpolation=TF.InterpolationMode.NEAREST)

        # --- colour jitter (RGB only, leave alpha alone) ---
        # Temporarily back to [0, 1] for torchvision jitter
        rgb = (rgb + 1.0) / 2.0
        rgb = self.color_jitter(rgb)
        rgb = rgb * 2.0 - 1.0

        return torch.cat([rgb, alpha], dim=0)   # (4, H, W)


# ─────────────────────────────────────────────────────────────
#  NT-XENT CONTRASTIVE LOSS  (SimCLR)
# ─────────────────────────────────────────────────────────────

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    NT-Xent loss for a batch of positive pairs (z1[i], z2[i]).

    z1, z2 : (B, D) L2-normalised projections
    For each sample i the positive pair is (z1[i], z2[i]).
    All other 2B-2 samples in the concatenated batch are negatives.
    """
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)                     # (2B, D)
    sim = torch.mm(z, z.T) / temperature                # (2B, 2B)

    # Mask out self-similarity on the diagonal
    mask = torch.eye(2 * B, device=z.device).bool()
    sim.masked_fill_(mask, float('-inf'))

    # Positive pairs: (i, i+B) and (i+B, i)
    labels = torch.cat([
        torch.arange(B, 2 * B, device=z.device),
        torch.arange(0, B,     device=z.device),
    ])

    return F.cross_entropy(sim, labels)


# ─────────────────────────────────────────────────────────────
#  CONTRASTIVE TRAINING DATASET
# ─────────────────────────────────────────────────────────────

class PieceContrastiveDataset(torch.utils.data.Dataset):
    """Returns two augmented views of each puzzle piece for contrastive training."""

    def __init__(self, image_paths: List[Path], size: int = 128):
        self.paths   = image_paths
        self.augment = PieceAugmentation(size=size)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = load_rgba(self.paths[idx])
        return self.augment(img)


# ─────────────────────────────────────────────────────────────
#  ENCODER TRAINING
# ─────────────────────────────────────────────────────────────

def train_encoder(
    image_paths : List[Path],
    embed_dim   : int,
    epochs      : int,
    lr          : float,
    batch_size  : int,
    temperature : float,
    checkpoint  : Path,
    device      : str,
) -> PieceEncoder:

    print(f"\n{'='*60}")
    print(f"  Training PieceEncoder  ({len(image_paths)} pieces)")
    print(f"  Epochs: {epochs}  |  LR: {lr}  |  Batch: {batch_size}")
    print(f"  Embed dim: {embed_dim}  |  Temperature: {temperature}")
    print(f"{'='*60}\n")

    dataset = PieceContrastiveDataset(image_paths)
    loader  = torch.utils.data.DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = 4,
        pin_memory  = True,
        drop_last   = True,
    )

    model     = PieceEncoder(embed_dim=embed_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for view1, view2 in loader:
            view1, view2 = view1.to(device), view2.to(device)

            z1 = model.forward_proj(view1)
            z2 = model.forward_proj(view2)
            loss = nt_xent_loss(z1, z2, temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg = epoch_loss / len(loader)

        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), checkpoint)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}/{epochs}  loss={avg:.4f}  best={best_loss:.4f}")

    print(f"\n  Best loss: {best_loss:.4f}  →  saved to {checkpoint}\n")

    # Reload best checkpoint
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model


# ─────────────────────────────────────────────────────────────
#  INFERENCE — extract embeddings
# ─────────────────────────────────────────────────────────────

def embed_images(image_paths: List[Path], model: PieceEncoder, device: str,
                 batch_size: int = 64) -> np.ndarray:
    """
    Extract (N, embed_dim) embeddings for a list of piece image paths.
    Runs model.forward() (inference path, not projection head).
    """
    model.eval()
    results = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        tensors = torch.stack([load_rgba(p) for p in batch_paths]).to(device)
        with torch.no_grad():
            embs = model(tensors)
        results.append(embs.cpu().numpy())

    return np.concatenate(results, axis=0)


# ─────────────────────────────────────────────────────────────
#  PATH COLLECTION  (mirrors fourier_descriptor.py)
# ─────────────────────────────────────────────────────────────

def collect_image_paths(root: Path, glob_pattern: str = "*/pieces/piece_*.png") -> List[Path]:
    if root.is_file():
        return [root]
    paths = sorted(root.rglob(glob_pattern))
    if not paths:
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"):
            paths.extend(sorted(root.rglob(ext)))
    return paths


# ─────────────────────────────────────────────────────────────
#  DATASET CREATION
# ─────────────────────────────────────────────────────────────

def create_dataset(
    root        : Path,
    glob_pattern: str,
    embed_dim   : int,
    output_dir  : Path,
    model       : PieceEncoder,
    device      : str,
) -> None:
    all_paths = collect_image_paths(root, glob_pattern)
    if not all_paths:
        print("[ERROR] No images found.", file=sys.stderr)
        sys.exit(1)

    # Group by parent directory (one group = one puzzle)
    puzzles: Dict[Path, List[Path]] = {}
    for p in all_paths:
        puzzles.setdefault(p.parent, []).append(p)

    print(f"\nFound {len(puzzles)} puzzle director{'y' if len(puzzles)==1 else 'ies'}.\n")

    X_rows, skipped = [], 0

    for puzzle_dir, piece_paths in sorted(puzzles.items()):
        piece_paths = sorted(piece_paths)
        try:
            embs = embed_images(piece_paths, model, device)   # (N, embed_dim)
        except Exception as e:
            print(f"  [WARN] Skipping {puzzle_dir.name}: {e}")
            skipped += 1
            continue

        n = len(embs)
        X_rows.append(embs.flatten())
        print(f"  ✓ {puzzle_dir.name:<30}  {n} pieces  →  X shape ({n * embed_dim},)")

    if not X_rows:
        print("[ERROR] No puzzles were successfully embedded.", file=sys.stderr)
        sys.exit(1)

    max_x = max(r.shape[0] for r in X_rows)
    X = np.zeros((len(X_rows), max_x), dtype=np.float32)
    for i, xr in enumerate(X_rows):
        X[i, :xr.shape[0]] = xr

    output_dir.mkdir(parents=True, exist_ok=True)
    x_path = output_dir / "convolution_curved_32.npy"
    np.save(x_path, X)

    print(f"\n{'='*60}")
    print(f"  Embeddings saved → {x_path}  ({x_path.stat().st_size / 1024:.1f} KB)")
    print(f"  Puzzles saved    : {len(X_rows)}")
    print(f"  Puzzles skipped  : {skipped}")
    print(f"  Embedding size   : {embed_dim} per piece")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = DATASET_ROOT
    if not root.exists():
        print(f"[ERROR] Path does not exist: {root}", file=sys.stderr)
        sys.exit(1)

    device     = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = Path(CHECKPOINT_PATH)

    # ── Step 1: Train or load encoder ────────────────────────
    if TRAIN_ENCODER:
        all_paths = collect_image_paths(root, GLOB_PATTERN)
        if not all_paths:
            print("[ERROR] No images found for training.", file=sys.stderr)
            sys.exit(1)
        print(f"[INFO] Found {len(all_paths)} piece images for encoder training.")
        model = train_encoder(
            image_paths = all_paths,
            embed_dim   = EMBEDDING_SIZE,
            epochs      = ENCODER_EPOCHS,
            lr          = ENCODER_LR,
            batch_size  = ENCODER_BATCH,
            temperature = TEMPERATURE,
            checkpoint  = checkpoint,
            device      = device,
        )
    else:
        print(f"[INFO] Loading encoder weights from {checkpoint}")
        model = PieceEncoder(embed_dim=EMBEDDING_SIZE).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device))

    model.eval()

    # ── Step 2: Extract embeddings and save ──────────────────
    if CREATE_DATASET:
        create_dataset(
            root         = root,
            glob_pattern = GLOB_PATTERN,
            embed_dim    = EMBEDDING_SIZE,
            output_dir   = OUTPUT_DIR,
            model        = model,
            device       = device,
        )