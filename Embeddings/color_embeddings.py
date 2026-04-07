"""
Color K-Means Embedding Extractor for Puzzle Pieces
=====================================================
Extracts dominant color embeddings from the non-transparent
pixels of each puzzle piece using MiniBatchKMeans.

Each piece → K_CLUSTERS centroids sorted by brightness → flattened
→ resized to exactly EMBEDDING_SIZE via linear interpolation → L2 normalized

Output dim per piece = EMBEDDING_SIZE  (any value, no restrictions)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

import cv2
import numpy as np
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans


def get_color_embedding(image_path: str, embedding_size: int = 96, k: int = 32) -> np.ndarray | None:
    """
    Extract a color embedding from a puzzle piece image.

    Args:
        image_path:     Path to an RGBA PNG puzzle piece.
        embedding_size: Output embedding dimension — any value, no restrictions.
        k:              Number of internal k-means clusters. Controls color
                        coverage before resampling. Not tied to embedding_size.

    Returns:
        np.ndarray of shape (embedding_size,) — brightness-sorted centroids
        flattened, interpolated to embedding_size, and L2 normalized.
        Returns None if the image cannot be processed.
    """
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None or img.shape[2] < 4:
        return None

    mask = img[:, :, 3] > 0
    if not np.any(mask):
        return None

    pixels = img[:, :, :3][mask].astype(np.float32)  # (M, 3) non-transparent pixels

    if len(pixels) > 5000:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pixels), 5000, replace=False)
        pixels = pixels[idx]

    kmeans = MiniBatchKMeans(n_clusters=k, n_init=3, random_state=42, max_iter=300)
    kmeans.fit(pixels)

    centroids = kmeans.cluster_centers_                          # (k, 3)

    brightness = np.sum(centroids, axis=1)
    sorted_centroids = centroids[np.argsort(brightness)]        # (k, 3)

    raw = sorted_centroids.flatten().astype(np.float32)         # (k*3,)

    # Resize to exactly embedding_size via linear interpolation —
    # no restriction on embedding_size whatsoever
    embedding = np.interp(
        np.linspace(0, len(raw) - 1, embedding_size),
        np.arange(len(raw)),
        raw,
    ).astype(np.float32)                                        # (embedding_size,)

    norm = np.linalg.norm(embedding)
    if norm > 1e-6:
        embedding /= norm

    return embedding


def create_dataset(root: Path, embedding_size: int, k: int,
                   glob_pattern: str, output_dir: Path) -> None:
    """
    Build a dataset from a directory of puzzle directories.

    Each puzzle directory becomes one training sample:
      - X row: all piece embeddings concatenated → (N_pieces * embedding_size,)

    Saves to:
      <output_dir>/color_curved_<embedding_size>.npy
    """
    all_paths = sorted(root.rglob(glob_pattern))
    if not all_paths:
        print("[ERROR] No images found.", file=sys.stderr)
        sys.exit(1)

    puzzles: dict[Path, list[Path]] = {}
    for p in all_paths:
        puzzles.setdefault(p.parent, []).append(p)

    print(f"\nFound {len(puzzles)} puzzle directories.\n")
    print(f"[INFO] k = {k} internal clusters  →  output embedding dim = {embedding_size}\n")

    X_rows = []
    skipped = 0

    for puzzle_dir, piece_paths in sorted(puzzles.items()):
        piece_paths = sorted(piece_paths)
        embeddings = []
        failed = False

        for piece_path in piece_paths:
            emb = get_color_embedding(str(piece_path),
                                      embedding_size=embedding_size, k=k)
            if emb is None:
                print(f"  [WARN] Could not embed {piece_path.name} — skipping {puzzle_dir.name}")
                failed = True
                break
            embeddings.append(emb)

        if failed:
            skipped += 1
            continue

        n = len(embeddings)
        X_rows.append(np.concatenate(embeddings))
        print(f"  ✓ {puzzle_dir.name:<30}  {n} pieces  →  ({n * embedding_size},)")

    if not X_rows:
        print("[ERROR] No puzzles were successfully embedded.", file=sys.stderr)
        sys.exit(1)

    max_len = max(r.shape[0] for r in X_rows)
    X = np.zeros((len(X_rows), max_len), dtype=np.float32)
    for i, xr in enumerate(X_rows):
        X[i, :xr.shape[0]] = xr

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"color_curved_{embedding_size}.npy"
    np.save(out_path, X)

    print(f"\n{'='*60}")
    print(f"  Saved → {out_path}  ({out_path.stat().st_size / 1024:.1f} KB)")
    print(f"  Shape  : {X.shape}")
    print(f"  Embedding dim per piece : {embedding_size}")
    print(f"  Internal k-means k      : {k}")
    print(f"  Puzzles saved  : {len(X_rows)}")
    print(f"  Puzzles skipped: {skipped}")
    print(f"{'='*60}")


if __name__ == "__main__":
    root = DATASET_ROOT
    if not root.exists():
        print(f"[ERROR] Path does not exist: {root}", file=sys.stderr)
        sys.exit(1)

    if CREATE_DATASET:
        create_dataset(
            root           = root,
            embedding_size = EMBEDDING_SIZE,
            k              = K_CLUSTERS,
            glob_pattern   = GLOB_PATTERN,
            output_dir     = OUTPUT_DIR,
        )