"""
Convolutional Embeddings Extractor (PyTorch) — Fixed Gabor Filter Bank
=======================================================================
Extracts shape embeddings from puzzle piece images using a fixed Gabor
filter bank. Zero learnable parameters — the kernels never change.

Configure the script by editing the GLOBAL CONFIGURATION section below.
"""

# ===========================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

import sys
import numpy as np
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


# ─────────────────────────────────────────────────────────────
#  GABOR FILTER GENERATION
# ─────────────────────────────────────────────────────────────

def gabor_kernel(
    size: int,
    sigma: float,
    theta: float,
    lambd: float,
    gamma: float = 0.5,
) -> np.ndarray:
    half = size // 2
    y, x = np.mgrid[-half:half+1, -half:half+1]

    x_rot =  x * np.cos(theta) + y * np.sin(theta)
    y_rot = -x * np.sin(theta) + y * np.cos(theta)

    envelope = np.exp(-0.5 * (x_rot**2 + (gamma * y_rot)**2) / sigma**2)
    carrier  = np.cos(2 * np.pi * x_rot / lambd)

    kernel = (envelope * carrier).astype(np.float32)
    kernel -= kernel.mean()
    std = kernel.std()
    if std > 1e-6:
        kernel /= std
    return kernel


def build_gabor_bank(
    in_channels: int,
    kernel_size: int = 7,
    orientations: int = 8,
    scales: int = 4,
    sigma_base: float = 2.0,
    lambd_base: float = 4.0,
) -> torch.Tensor:
    kernels = []
    thetas  = [np.pi * i / orientations for i in range(orientations)]
    for scale_idx in range(scales):
        sigma = sigma_base * (2 ** scale_idx)
        lambd = lambd_base * (2 ** scale_idx)
        for theta in thetas:
            kernels.append(gabor_kernel(kernel_size, sigma, theta, lambd))

    bank = np.stack(kernels, axis=0)                              # (O*S, kH, kW)
    bank = np.tile(bank[:, np.newaxis], (in_channels, 1, 1, 1))  # (O*S*C, 1, kH, kW)
    return torch.from_numpy(bank)


# ─────────────────────────────────────────────────────────────
#  GABOR CONV LAYER
# ─────────────────────────────────────────────────────────────

class GaborConv2d(nn.Module):
    """Depthwise conv with fixed Gabor kernels. Zero learnable parameters."""

    def __init__(self, in_channels: int, kernel_size: int = 7,
                 orientations: int = 8, scales: int = 4):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = in_channels * orientations * scales
        weight = build_gabor_bank(in_channels, kernel_size, orientations, scales)
        self.register_buffer("weight", weight)
        self.padding = kernel_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight, bias=None,
                        padding=self.padding, groups=self.in_channels)


# ─────────────────────────────────────────────────────────────
#  ENCODER
# ─────────────────────────────────────────────────────────────

class GaborEncoder(nn.Module):
    """
    Two-layer fixed Gabor encoder followed by a random orthogonal projection.
    Learnable parameters: 0.
    """

    def __init__(self, embed_dim: int = 64):
        super().__init__()

        self.gabor1 = GaborConv2d(in_channels=3, kernel_size=7, orientations=8, scales=4)
        self.gabor2 = GaborConv2d(in_channels=self.gabor1.out_channels,
                                  kernel_size=7, orientations=4, scales=2)

        proj = self._rand_proj(self.gabor2.out_channels, embed_dim)
        self.register_buffer("proj", proj)
        self.pool = nn.AdaptiveAvgPool2d(1)

        assert sum(p.numel() for p in self.parameters()) == 0, \
            "Unexpected learnable parameters found."

    @staticmethod
    def _rand_proj(in_dim: int, out_dim: int) -> torch.Tensor:
        if in_dim <= out_dim:
            q, _ = torch.linalg.qr(torch.randn(out_dim, in_dim))
            proj = torch.zeros(in_dim, out_dim)
            proj[:, :in_dim] = q[:in_dim].T
        else:
            q, _ = torch.linalg.qr(torch.randn(in_dim, out_dim))
            proj = q
        return proj.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(F.max_pool2d(self.gabor1(x), 2))
        x = F.relu(F.max_pool2d(self.gabor2(x), 2))
        x = self.pool(x).flatten(1)
        x = x @ self.proj
        return F.normalize(x, p=2, dim=1)


# ─────────────────────────────────────────────────────────────
#  IMAGE LOADING
# ─────────────────────────────────────────────────────────────

_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def embed_images(image_paths: List[Path], model: nn.Module) -> np.ndarray:
    """Return (N, embed_dim) array for a list of image paths."""
    tensors = torch.stack([
        _transform(Image.open(p).convert("RGB")) for p in image_paths
    ])
    with torch.no_grad():
        return model(tensors).cpu().numpy()


# ─────────────────────────────────────────────────────────────
#  PATH COLLECTION  (mirrors fourier_descriptor.py)
# ─────────────────────────────────────────────────────────────

def collect_image_paths(
    root: Path,
    glob_pattern: str = "*/pieces/piece_*.png",
) -> List[Path]:
    if root.is_file():
        return [root]
    paths = sorted(root.rglob(glob_pattern))
    if not paths:
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"):
            paths.extend(sorted(root.rglob(ext)))
    return paths


# ─────────────────────────────────────────────────────────────
#  DATASET CREATION  (mirrors fourier_descriptor.py)
# ─────────────────────────────────────────────────────────────

def create_dataset(
    root: Path,
    glob_pattern: str,
    embed_dim: int,
    output_dir: Path,
) -> None:
    """
    Build a dataset from a directory of puzzle directories.

      X row : piece embeddings concatenated into a flat vector  (N * embed_dim,)

    Saves to:
      <output_dir>/gabor.npy — embeddings
    """
    all_paths = collect_image_paths(root, glob_pattern)
    if not all_paths:
        print("[ERROR] No images found.", file=sys.stderr)
        sys.exit(1)

    # Group by immediate parent (one group = one puzzle)
    puzzles: Dict[Path, List[Path]] = {}
    for p in all_paths:
        puzzles.setdefault(p.parent, []).append(p)

    print(f"\nFound {len(puzzles)} puzzle director{'y' if len(puzzles) == 1 else 'ies'}.\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = GaborEncoder(embed_dim=embed_dim).to(device).eval()
    print(f"[INFO] Device            : {device}")
    print(f"[INFO] Embedding size    : {embed_dim}")
    print(f"[INFO] Fixed buffer vals : {sum(b.numel() for b in model.buffers())}  "
          f"(Gabor kernels + projection)\n")

    X_rows = []
    skipped = 0

    for puzzle_dir, piece_paths in sorted(puzzles.items()):
        piece_paths = sorted(piece_paths)
        try:
            embs = embed_images(piece_paths, model)  # (N, embed_dim)
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

    # Pad rows to uniform length (0.0 padding)
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


def load_dataset(x_path: Path) -> List[np.ndarray]:
    """Load a dataset saved by create_dataset(), stripping zero-padding."""
    X_pad = np.load(x_path)
    return [row[row != 0.0] for row in X_pad]


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = DATASET_ROOT
    if not root.exists():
        print(f"[ERROR] Path does not exist: {root}", file=sys.stderr)
        sys.exit(1)

    if not CREATE_DATASET:
        print("[INFO] Set CREATE_DATASET = True to generate and save embeddings.")
        sys.exit(0)

    create_dataset(
        root         = root,
        glob_pattern = GLOB_PATTERN,
        embed_dim    = EMBEDDING_SIZE,
        output_dir   = Path(OUTPUT_DIR),
    )