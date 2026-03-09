"""
Fourier Descriptor Embedding Extractor for Puzzle Pieces
=========================================================
Extracts shape-based Fourier descriptor embeddings from puzzle piece images.

Configure the script by editing the GLOBAL CONFIGURATION section below.
"""

# ===========================================================================
# GLOBAL CONFIGURATION — edit these values instead of using command-line args
# ===========================================================================

# Path to a single image file or a root directory containing puzzle pieces.
INPUT = "/home/hruday/studies/computer_vision/puzzle_solver/Puzzle-solver/Dataset/train_set_curved"

# Number of Fourier coefficients to keep (embedding dimensionality).
EMBEDDING_SIZE = 64

# Directory where dataset .npy files will be saved (used only when CREATE_DATASET=True).
# Files are always named  fourier.npy  and  fourier_target.npy  inside this directory.
OUTPUT_DIR = "/home/hruday/studies/computer_vision/puzzle_solver/Puzzle-solver/Dataset"  # e.g. "/path/to/output_dir"

# Recursively search subdirectories for images.
RECURSIVE = True

# Glob pattern relative to INPUT for image discovery.
GLOB_PATTERN = "*/pieces/piece_*.png"

# ── Rotation invariance test ─────────────────────────────────────────────────
# Set TEST_ROTATION = True to run a rotation-invariance test instead of
# computing embeddings.
TEST_ROTATION = False

# Number of images to use for the rotation test.
TEST_N = 5

# ── Sampling ─────────────────────────────────────────────────────────────────
# Take only the first SAMPLE images to embed. Set to None to use all images.
# Use this to only inspect the embeddings - does not control the size of the dataset created
SAMPLE = 200  # e.g. 100

# ── Dataset creation ─────────────────────────────────────────────────────────
# Set CREATE_DATASET = True to build a training dataset where each puzzle
# directory becomes one sample (piece embeddings concatenated → X, 0-based
# piece indices → y). Saves to OUTPUT_DIR as fourier.npy / fourier_target.npy.
CREATE_DATASET = True

# ===========================================================================


import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.distance import cosine, euclidean


def get_fourier_descriptors(image_path, embedding_size=64):
    img_bgra = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    thresh = img_bgra[:, :, 3]

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    contour_complex = contour[:, 0, 0] + 1j * contour[:, 0, 1]

    descriptors = np.fft.fft(contour_complex)
    descriptors = descriptors[1:]

    magnitudes = np.abs(descriptors)
    if magnitudes[0] != 0:
        magnitudes = magnitudes / magnitudes[0]

    return magnitudes[:embedding_size]


def embed_image(
    image_path: str | Path,
    embedding_size: int = 64,
) -> np.ndarray | None:
    """
    Load an image and return its Fourier descriptor embedding.

    Returns None if the image cannot be processed.
    """
    return get_fourier_descriptors(image_path=str(image_path), embedding_size=embedding_size)


# ---------------------------------------------------------------------------
# Batch Processing
# ---------------------------------------------------------------------------

def collect_image_paths(
    root: Path,
    recursive: bool = True,
    glob_pattern: str = "*/pieces/piece_*.png",
) -> list[Path]:
    """Return all matching image paths under `root`."""
    if root.is_file():
        return [root]
    fn = root.rglob if recursive else root.glob
    paths = sorted(fn(glob_pattern))
    # Fallback: accept any common image extension
    if not paths:
        exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff")
        for ext in exts:
            paths.extend(sorted(fn(ext)))
    return paths


def batch_embed(
    image_paths: list[Path],
    embedding_size: int = 64,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """
    Compute Fourier descriptor embeddings for a list of image paths.

    Returns a dict mapping image path strings -> embedding arrays.
    """
    results: dict[str, np.ndarray] = {}
    total = len(image_paths)

    for i, path in enumerate(image_paths, 1):
        if verbose:
            print(f"[{i:>4}/{total}] Processing: {path}")
        emb = embed_image(path, embedding_size=embedding_size)
        if emb is not None:
            results[str(path)] = emb

    return results


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_embeddings(embeddings: dict[str, np.ndarray], output_path: Path) -> None:
    """Save embeddings dict to a compressed .npz file."""
    keys = list(embeddings.keys())
    vectors = np.stack(list(embeddings.values()))  # (N, embedding_size)

    np.savez_compressed(
        output_path,
        paths=np.array(keys, dtype=object),
        embeddings=vectors,
    )
    print(f"\n✓ Saved {len(keys)} embeddings -> {output_path}")
    print(f"  Embedding matrix shape : {vectors.shape}")
    print(f"  File size              : {output_path.stat().st_size / 1024:.1f} KB")


def load_embeddings(npz_path: Path) -> dict[str, np.ndarray]:
    """Load embeddings previously saved with `save_embeddings`."""
    data = np.load(npz_path, allow_pickle=True)
    return {str(p): v for p, v in zip(data["paths"], data["embeddings"])}


# ---------------------------------------------------------------------------
# Dataset Creation
# ---------------------------------------------------------------------------

def create_dataset(
    root: Path,
    embedding_size: int,
    glob_pattern: str,
    output_dir: Path,
) -> None:
    """
    Build a training dataset from a directory of puzzle directories.

    Each puzzle directory becomes one training sample:
      - X row : all piece embeddings concatenated into a single flat vector,
                sorted by filename so ordering is deterministic.
      - y row : 0-based index vector [0, 1, ..., num_pieces-1].

    Always saves to:
      <output_dir>/fourier.npy        (embeddings)
      <output_dir>/fourier_target.npy (targets)

    Puzzles where any piece fails to embed are skipped with a warning.
    """
    all_piece_paths = collect_image_paths(root, recursive=True, glob_pattern=glob_pattern)
    if not all_piece_paths:
        print("[ERROR] No images found under the given glob pattern.", file=sys.stderr)
        sys.exit(1)

    # Group pieces by their immediate parent directory (one group = one puzzle)
    puzzles: dict[Path, list[Path]] = {}
    for p in all_piece_paths:
        puzzles.setdefault(p.parent, []).append(p)

    print(f"\nFound {len(puzzles)} puzzle director{'y' if len(puzzles) == 1 else 'ies'}.\n")

    X_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    puzzle_dirs: list[str] = []
    skipped = 0

    for puzzle_dir, piece_paths in sorted(puzzles.items()):
        piece_paths = sorted(piece_paths)  # deterministic order by filename
        embeddings = []
        failed = False

        for piece_path in piece_paths:
            emb = embed_image(piece_path, embedding_size=embedding_size)
            if emb is None:
                print(f"  [WARN] Could not embed {piece_path.name} — skipping puzzle {puzzle_dir.name}")
                failed = True
                break
            embeddings.append(emb)

        if failed:
            skipped += 1
            continue

        n = len(embeddings)
        # Shuffle piece order, applying the same permutation to both X and y
        perm = np.random.permutation(n)
        embeddings = [embeddings[i] for i in perm]
        X_rows.append(np.concatenate(embeddings))               # flat vector: n * embedding_size
        y_rows.append(np.arange(0, n, dtype=np.int32)[perm])    # e.g. [2, 4, 1, 3, ...]
        puzzle_dirs.append(str(puzzle_dir))
        print(f"  ✓ {puzzle_dir.name:<30}  {n} pieces  →  X shape ({n * embedding_size},)")

    if not X_rows:
        print("[ERROR] No puzzles were successfully embedded.", file=sys.stderr)
        sys.exit(1)

    # Pad rows to uniform length (0.0 for X, -1 for y)
    max_x_len = max(xr.shape[0] for xr in X_rows)
    max_y_len = max(yr.shape[0] for yr in y_rows)

    X = np.zeros((len(X_rows), max_x_len), dtype=np.float64)
    y = np.full((len(y_rows), max_y_len), fill_value=-1, dtype=np.int32)
    for i, (xr, yr) in enumerate(zip(X_rows, y_rows)):
        X[i, :xr.shape[0]] = xr
        y[i, :yr.shape[0]] = yr

    output_dir.mkdir(parents=True, exist_ok=True)
    x_path = output_dir / "fourier.npy"
    y_path = output_dir / "fourier_target.npy"

    np.save(x_path, X)
    np.save(y_path, y)

    print(f"\n{'='*60}")
    print(f"  Embeddings saved -> {x_path}  ({x_path.stat().st_size / 1024:.1f} KB)")
    print(f"  Targets saved    -> {y_path}  ({y_path.stat().st_size / 1024:.1f} KB)")
    print(f"  Puzzles saved    : {len(X_rows)}")
    print(f"  Puzzles skipped  : {skipped}")
    print(f"  Embedding size   : {embedding_size} per piece")
    print(f"{'='*60}")


def load_dataset(
    x_path: Path,
    y_path: Path,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Load a dataset saved by create_dataset().

    Returns:
        X : list of 1-D float arrays (one flat vector per puzzle)
        y : list of 1-D int arrays   (0-based piece indices per puzzle)
    """
    X_pad = np.load(x_path)   # shape (N, max_x_len), padded with 0.0
    y_pad = np.load(y_path)   # shape (N, max_y_len), padded with -1
    X = [row[row != 0.0] for row in X_pad]
    y = [row[row != -1]  for row in y_pad]
    return X, y


# ---------------------------------------------------------------------------
# Rotation Invariance Test
# ---------------------------------------------------------------------------

def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image by `angle` degrees around its centre, expanding canvas to avoid clipping."""
    h, w = image.shape[:2]
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += new_w / 2 - cx
    M[1, 2] += new_h / 2 - cy

    return cv2.warpAffine(image, M, (new_w, new_h),
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(255, 255, 255, 0))


def test_rotation_invariance(
    image_paths: list[Path],
    n: int,
    embedding_size: int,
) -> None:
    selected_paths = image_paths[:n]

    rotations = {
        "90 CW":  cv2.ROTATE_90_CLOCKWISE,
        "180":    cv2.ROTATE_180,
        "90 CCW": cv2.ROTATE_90_COUNTERCLOCKWISE,
    }

    print(f"{'Image':<15} | {'Rotation':<10} | {'L2 Distance':<12} | {'Cosine Sim'}")
    print("-" * 55)

    for path in selected_paths:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue

        embed = embed_image(path, embedding_size=embedding_size)
        if embed is None:
            continue

        for label, rot_code in rotations.items():
            rotated_img = cv2.rotate(img, rot_code)
            tmp_path = Path(f"/tmp/_rotation_test_{path.stem}_{label.replace(' ', '_')}.png")
            cv2.imwrite(str(tmp_path), rotated_img)

            embed_rotated = embed_image(tmp_path, embedding_size=embedding_size)
            if embed_rotated is None:
                print(f"{path.name[:15]:<15} | {label:<10} | {'ERROR':>12} | ERROR")
                continue

            l2_dist = euclidean(embed, embed_rotated)
            cos_sim = 1 - cosine(embed, embed_rotated)

            print(f"{path.name[:15]:<15} | {label:<10} | {l2_dist:<12.4f} | {cos_sim:.4f}")


# ---------------------------------------------------------------------------
# Programmatic API (import-friendly)
# ---------------------------------------------------------------------------

__all__ = [
    "embed_image",
    "batch_embed",
    "rotate_image",
    "test_rotation_invariance",
    "save_embeddings",
    "load_embeddings",
    "create_dataset",
    "load_dataset",
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    root = Path(INPUT)
    if not root.exists():
        print(f"[ERROR] Path does not exist: {root}", file=sys.stderr)
        sys.exit(1)

    embedding_size = EMBEDDING_SIZE
    if embedding_size % 2 != 0:
        embedding_size += 1
        print(f"[INFO] EMBEDDING_SIZE rounded up to {embedding_size} (must be even).")

    if CREATE_DATASET:
        create_dataset(
            root=root,
            embedding_size=embedding_size,
            glob_pattern=GLOB_PATTERN,
            output_dir=Path(OUTPUT_DIR),
        )
        sys.exit(0)

    image_paths = collect_image_paths(root, recursive=RECURSIVE, glob_pattern=GLOB_PATTERN)
    if not image_paths:
        print("[ERROR] No images found.", file=sys.stderr)
        sys.exit(1)

    if TEST_ROTATION:
        test_rotation_invariance(image_paths, n=TEST_N, embedding_size=embedding_size)
    else:
        if SAMPLE is not None:
            total = len(image_paths)
            n = min(SAMPLE, total)
            image_paths = image_paths[:n]
            print(f"[INFO] Sampled {n}/{total} images.")

        print(f"\n{'='*60}")
        print(f"  Fourier Descriptor Extractor")
        print(f"  Input          : {root}")
        print(f"  Embedding size : {embedding_size}")
        print(f"  Glob pattern   : {GLOB_PATTERN}")
        print(f"{'='*60}\n")

        print(f"Found {len(image_paths)} image(s) to embed.\n")

        embeddings = batch_embed(image_paths, embedding_size=embedding_size)

        print(f"\n✓ Successfully embedded {len(embeddings)}/{len(image_paths)} images.\n")