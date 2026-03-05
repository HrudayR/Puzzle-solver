"""
Convolutional Embeddings Extractor (PyTorch) — Gabor Filter Bank
=================================================================
Uses a fixed Gabor filter bank with zero learnable parameters to extract
embeddings from PNG images.

Architecture:
    GaborConv (frozen)  →  ReLU  →  MaxPool
    GaborConv (frozen)  →  ReLU  →  MaxPool
    AdaptiveAvgPool(1×1) → Flatten → L2-normalise → (B, embed_dim)

  embed   – Extract per-directory concatenated embeddings with target vectors.
  rotate  – Compare per-image embeddings before and after random rotation.

Usage:
    python conv_embeddings.py embed \\
        --image_dirs "/home/hruday/.../train_set_curved/*/pieces" \\
        --embedding_size 256 --output embeddings.npy --save_csv

    python conv_embeddings.py rotate \\
        --image_dirs "/home/hruday/.../train_set_curved/*/pieces" \\
        --num_dirs 5 --angle_range 15 345 --embedding_size 256

Install:
    pip install torch torchvision Pillow numpy


python conv_embeddings.py rotate --image_dirs "/home/hruday/studies/computer_vision/puzzle_solver/Puzzle-solver/Dataset/train_set_curved/*/pieces" --num_dirs 5 --embedding_size 32
"""

import argparse
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    psi: float = 0.0,
) -> np.ndarray:
    """
    Generate a single 2-D Gabor filter kernel.

    Args:
        size  : Kernel height and width (pixels). Should be odd.
        sigma : Standard deviation of the Gaussian envelope.
                Controls the filter's spatial extent.
        theta : Orientation of the filter's normal (radians).
                e.g. 0=horizontal, π/4=diagonal, π/2=vertical.
        lambd : Wavelength of the sinusoidal carrier (pixels).
                Smaller = higher frequency / finer detail.
        gamma : Spatial aspect ratio. 0.5 keeps the filter slightly
                elongated along its preferred orientation.
        psi   : Phase offset of the sinusoid (radians).

    Returns:
        kernel : np.ndarray of shape (size, size), dtype float32,
                 normalised to zero mean and unit std.
    """
    half = size // 2
    y, x = np.mgrid[-half:half+1, -half:half+1]

    # Rotate coordinate system by theta
    x_rot =  x * np.cos(theta) + y * np.sin(theta)
    y_rot = -x * np.sin(theta) + y * np.cos(theta)

    # Gaussian envelope
    envelope = np.exp(
        -0.5 * (x_rot**2 + (gamma * y_rot)**2) / sigma**2
    )
    # Sinusoidal carrier
    carrier = np.cos(2 * np.pi * x_rot / lambd + psi)

    kernel = (envelope * carrier).astype(np.float32)

    # Normalise: zero mean, unit std — prevents activation scale drift
    kernel -= kernel.mean()
    std = kernel.std()
    if std > 1e-6:
        kernel /= std

    return kernel   # (size, size)


def build_gabor_bank(
    in_channels: int,
    kernel_size: int = 7,
    orientations: int = 8,
    scales: int = 4,
    sigma_base: float = 2.0,
    lambd_base: float = 4.0,
) -> torch.Tensor:
    """
    Build a bank of Gabor filters covering multiple orientations and scales.

    Each (orientation, scale) pair produces one filter per input channel,
    giving a total of (orientations × scales × in_channels) output channels.

    Args:
        in_channels   : Number of input channels (1=grayscale, 3=RGB).
        kernel_size   : Spatial size of each filter (should be odd).
        orientations  : Number of evenly-spaced orientations in [0, π).
        scales        : Number of frequency scales (each doubles the wavelength).
        sigma_base    : Sigma for the finest scale; doubles per scale level.
        lambd_base    : Wavelength for the finest scale; doubles per scale level.

    Returns:
        weight : torch.Tensor of shape
                 (orientations * scales * in_channels, 1, kernel_size, kernel_size)
                 ready to use as a depthwise conv weight (groups=in_channels).

        Actually returns shape (out_ch, 1, kH, kW) for use with groups=in_channels,
        where out_ch = orientations * scales * in_channels.
    """
    kernels = []
    thetas  = [np.pi * i / orientations for i in range(orientations)]

    for scale_idx in range(scales):
        sigma = sigma_base * (2 ** scale_idx)
        lambd = lambd_base * (2 ** scale_idx)
        for theta in thetas:
            k = gabor_kernel(kernel_size, sigma=sigma, theta=theta, lambd=lambd)
            kernels.append(k)

    # Each kernel is applied to every input channel independently (depthwise),
    # so we tile the kernel bank across in_channels.
    # Final shape: (orientations*scales*in_channels, 1, kH, kW)
    bank = np.stack(kernels, axis=0)                       # (O*S, kH, kW)
    bank = np.tile(bank[:, np.newaxis], (in_channels, 1, 1, 1))
    #                                    ^ repeat for each input channel
    # shape: (O*S*in_channels, 1, kH, kW)

    return torch.from_numpy(bank)


# ─────────────────────────────────────────────────────────────
#  GABOR CONV LAYER
# ─────────────────────────────────────────────────────────────

class GaborConv2d(nn.Module):
    """
    A convolutional layer whose kernels are fixed Gabor filters.
    Zero learnable parameters — weights are frozen at construction.

    Operates as a depthwise convolution: each input channel is
    filtered independently by every (orientation, scale) kernel,
    then the results are concatenated across channels.

    Input  : (B, in_channels, H, W)
    Output : (B, in_channels * orientations * scales, H', W')
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 7,
        orientations: int = 8,
        scales: int = 4,
        sigma_base: float = 2.0,
        lambd_base: float = 4.0,
    ):
        super().__init__()

        self.in_channels  = in_channels
        self.kernel_size  = kernel_size
        self.orientations = orientations
        self.scales       = scales
        self.out_channels = in_channels * orientations * scales

        weight = build_gabor_bank(
            in_channels, kernel_size, orientations, scales,
            sigma_base, lambd_base,
        )   # (out_channels, 1, kH, kW)

        # Register as a buffer — stored in state_dict but NOT a learnable parameter
        self.register_buffer("weight", weight)
        self.padding = kernel_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Depthwise conv: groups=in_channels means each input channel
        # is convolved with its own subset of filters independently.
        return F.conv2d(
            x,
            self.weight,
            bias=None,
            padding=self.padding,
            groups=self.in_channels,
        )


# ─────────────────────────────────────────────────────────────
#  GABOR ENCODER
# ─────────────────────────────────────────────────────────────

class GaborEncoder(nn.Module):
    """
    Fixed Gabor filter bank encoder with ZERO learnable parameters.

    Architecture:
        GaborConv  (frozen)  →  ReLU  →  MaxPool(2)
        GaborConv  (frozen)  →  ReLU  →  MaxPool(2)
        AdaptiveAvgPool(1×1) →  Flatten  →  linear_proj (frozen)
        →  L2-normalise  →  (B, embed_dim)

    The linear projection at the end is also fixed (random orthogonal
    projection) — it only reduces dimensionality, it is not learned.

    Learnable parameters: 0
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 512,
        normalize: bool = True,
        kernel_size: int = 7,
        orientations: int = 8,
        scales: int = 4,
    ):
        """
        Args:
            in_channels  : Input image channels (3 for RGB, 1 for grayscale).
            embed_dim    : Output embedding dimension.
            normalize    : L2-normalize the final embedding.
            kernel_size  : Gabor kernel spatial size (odd number).
            orientations : Number of filter orientations per scale.
            scales       : Number of frequency scales.
        """
        super().__init__()
        self.normalize = normalize

        # ── Layer 1: Gabor bank ───────────────────────────────
        self.gabor1 = GaborConv2d(
            in_channels, kernel_size, orientations, scales,
        )
        ch1 = self.gabor1.out_channels   # in_channels * orientations * scales

        # ── Layer 2: second Gabor bank on the response map ───
        # Each channel from layer 1 is processed independently again.
        self.gabor2 = GaborConv2d(
            ch1, kernel_size, orientations=4, scales=2,
        )
        ch2 = self.gabor2.out_channels   # ch1 * 4 * 2

        # ── Fixed projection: ch2 → embed_dim ────────────────
        # Uses a random orthogonal matrix — no learning, just dimensionality
        # reduction. Registered as a buffer (not a parameter).
        proj = self._random_orthogonal_projection(ch2, embed_dim)
        self.register_buffer("proj", proj)   # (ch2, embed_dim)

        self.pool = nn.AdaptiveAvgPool2d(1)

        # ── Verify zero learnable params ──────────────────────
        n_learnable = sum(p.numel() for p in self.parameters())
        assert n_learnable == 0, (
            f"Expected 0 learnable parameters, found {n_learnable}. "
            "Check that no nn.Parameter was accidentally introduced."
        )

    @staticmethod
    def _random_orthogonal_projection(in_dim: int, out_dim: int) -> torch.Tensor:
        """
        Create a fixed random projection matrix using QR decomposition.
        The columns are orthonormal, giving a well-conditioned projection.
        Shape: (in_dim, out_dim)
        """
        if in_dim <= out_dim:
            # If projecting UP (rare), just pad with zeros
            proj = torch.zeros(in_dim, out_dim)
            q, _ = torch.linalg.qr(torch.randn(out_dim, in_dim))
            proj[:, :in_dim] = q[:in_dim].T
        else:
            q, _ = torch.linalg.qr(torch.randn(in_dim, out_dim))
            proj = q  # (in_dim, out_dim)
        return proj.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Layer 1
        x = self.gabor1(x)                  # (B, ch1, H, W)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)             # (B, ch1, H/2, W/2)

        # Layer 2
        x = self.gabor2(x)                  # (B, ch2, H/2, W/2)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)             # (B, ch2, H/4, W/4)

        # Global pool → flatten → (B, ch2)
        x = self.pool(x).flatten(1)

        # Fixed projection → (B, embed_dim)
        x = x @ self.proj

        if self.normalize:
            x = F.normalize(x, p=2, dim=1)

        return x


# ─────────────────────────────────────────────────────────────
#  DIRECTORY RESOLUTION
# ─────────────────────────────────────────────────────────────

def resolve_image_dirs(patterns: List[str]) -> List[Path]:
    dirs: List[Path] = []
    for pattern in patterns:
        matches = sorted(Path("/").glob(pattern.lstrip("/")))
        if not matches:
            p = Path(pattern)
            if p.is_dir():
                matches = [p]
            else:
                print(f"[WARN] No directories matched: {pattern}")
        dirs.extend(m for m in matches if m.is_dir())

    seen, unique = set(), []
    for d in dirs:
        if d not in seen:
            seen.add(d)
            unique.append(d)
    return unique


# ─────────────────────────────────────────────────────────────
#  DATA LOADING & TRANSFORMS
# ─────────────────────────────────────────────────────────────

def build_transform(
    image_size: Optional[Tuple[int, int]] = None,
    rotation_angle: Optional[float] = None,
) -> transforms.Compose:
    steps = []
    if image_size:
        steps.append(transforms.Resize(image_size))
    if rotation_angle is not None:
        steps.append(transforms.Lambda(
            lambda img: img.rotate(rotation_angle, expand=True)
        ))
    steps += [
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ]
    return transforms.Compose(steps)


def load_batch(image_paths: List[Path], transform: transforms.Compose) -> torch.Tensor:
    return torch.stack([transform(Image.open(p).convert("RGB")) for p in image_paths])


def embed_images(
    image_paths: List[Path],
    model: nn.Module,
    transform: transforms.Compose,
    batch_size: int,
    device: str,
) -> np.ndarray:
    """Embed images and return array of shape (num_images, embed_dim)."""
    all_embs = []
    for i in range(0, len(image_paths), batch_size):
        batch = load_batch(image_paths[i : i + batch_size], transform).to(device)
        with torch.no_grad():
            emb = model(batch)
        all_embs.append(emb.cpu().numpy())
    return np.vstack(all_embs)


# ─────────────────────────────────────────────────────────────
#  TARGET VECTOR
# ─────────────────────────────────────────────────────────────

def make_target_and_embedding(
    per_image_embs: np.ndarray,
    shuffle: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given per-image embeddings of shape (N, embed_dim), produce:
      - target    : [0, 1, ..., N-1] or shuffled permutation
      - embedding : (1, N * embed_dim) concatenated flat vector

    target[i] = original image index at position i in the embedding.
    """
    n = per_image_embs.shape[0]
    if shuffle:
        perm         = np.random.permutation(n)
        ordered_embs = per_image_embs[perm]
        target       = perm
    else:
        ordered_embs = per_image_embs
        target       = np.arange(n)

    embedding = ordered_embs.flatten()[np.newaxis, :]
    return target, embedding


# ─────────────────────────────────────────────────────────────
#  SIMILARITY METRICS
# ─────────────────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.flatten(), b.flatten()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a.flatten() - b.flatten()))


# ─────────────────────────────────────────────────────────────
#  TABLE HELPERS
# ─────────────────────────────────────────────────────────────

def truncate_path(path: str, max_len: int) -> str:
    if len(path) <= max_len:
        return path
    keep = (max_len - 3) // 2
    return path[:keep] + "..." + path[-(max_len - keep - 3):]


def print_table(rows: List[Tuple], path_col_width: int = 55) -> None:
    sep = "─" * (path_col_width + 42)
    print(sep)
    print(f"  {'Image Path':<{path_col_width}}  {'Angle':>7}  {'CosSim':>8}  {'L2 Dist':>10}")
    print(sep)
    prev_dir = None
    for dir_path, img_name, angle, cos_sim, l2_dist in rows:
        if dir_path != prev_dir:
            if prev_dir is not None:
                print(f"  {'·' * path_col_width}")
            prev_dir = dir_path
        display = truncate_path(f"{dir_path}/{img_name}", path_col_width)
        print(f"  {display:<{path_col_width}}  {angle:>6.1f}°  {cos_sim:>8.4f}  {l2_dist:>10.4f}")
    print(sep)


# ─────────────────────────────────────────────────────────────
#  SHARED MODEL BUILDER
# ─────────────────────────────────────────────────────────────

def build_model(
    embedding_size: int,
    normalize: bool,
    device: str,
    kernel_size: int = 7,
    orientations: int = 8,
    scales: int = 4,
) -> nn.Module:
    model = GaborEncoder(
        in_channels=3,
        embed_dim=embedding_size,
        normalize=normalize,
        kernel_size=kernel_size,
        orientations=orientations,
        scales=scales,
    ).to(device)
    model.eval()

    n_learnable = sum(p.numel() for p in model.parameters())
    n_fixed     = sum(b.numel() for b in model.buffers())
    print(f"[INFO] Learnable params  : {n_learnable}  ← always 0")
    print(f"[INFO] Fixed buffer vals : {n_fixed}  (Gabor kernels + projection)")
    return model


# ─────────────────────────────────────────────────────────────
#  MODE 1 — EMBED
# ─────────────────────────────────────────────────────────────

def run_embed(
    image_dirs: List[str],
    embedding_size: int = 512,
    batch_size: int = 16,
    image_size: Optional[Tuple[int, int]] = None,
    normalize: bool = True,
    shuffle: bool = False,
    num_dirs: Optional[int] = None,
    output_path: Optional[str] = None,
    save_csv: bool = False,
    device: Optional[str] = None,
    kernel_size: int = 7,
    orientations: int = 8,
    scales: int = 4,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract per-directory concatenated embeddings with matching target vectors.
    Uses a fixed Gabor filter bank — zero learnable parameters.

    Output:
      embeddings : (num_dirs, N * embedding_size)
      targets    : (num_dirs, N)
      dir_names  : list of directory names
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device            : {device}")

    dirs = resolve_image_dirs(image_dirs)
    if not dirs:
        raise FileNotFoundError(f"No valid directories found for: {image_dirs}")

    dir_images: Dict[Path, List[Path]] = {}
    for d in dirs:
        imgs = sorted(d.glob("*.png"))
        if not imgs:
            print(f"[WARN] No PNG files, skipping: {d}")
        else:
            dir_images[d] = imgs

    if not dir_images:
        raise FileNotFoundError("No PNG files found in any resolved directory.")

    # Optionally subsample directories
    if num_dirs is not None:
        if num_dirs > len(dir_images):
            print(f"[WARN] Requested {num_dirs} dirs but only {len(dir_images)} available. "
                  f"Using all {len(dir_images)}.")
        else:
            sampled_keys = random.sample(list(dir_images.keys()), num_dirs)
            dir_images   = {k: dir_images[k] for k in sampled_keys}

    counts        = {d: len(v) for d, v in dir_images.items()}
    unique_counts = set(counts.values())
    n_images      = next(iter(unique_counts))
    uniform       = len(unique_counts) == 1

    print(f"[INFO] Directories       : {len(dir_images)}")
    print(f"[INFO] Images/dir        : {n_images if uniform else 'variable'}")
    print(f"[INFO] Embedding dim     : {embedding_size}")
    print(f"[INFO] Gabor orientations: {orientations}")
    print(f"[INFO] Gabor scales      : {scales}")
    print(f"[INFO] Gabor kernel size : {kernel_size}")
    print(f"[INFO] Shuffle           : {shuffle}")
    if uniform:
        print(f"[INFO] Embedding out     : ({len(dir_images)}, {n_images * embedding_size})")
        print(f"[INFO] Target out        : ({len(dir_images)}, {n_images})")

    model     = build_model(embedding_size, normalize, device, kernel_size, orientations, scales)
    transform = build_transform(image_size)
    print()

    all_embeddings: List[np.ndarray] = []
    all_targets:    List[np.ndarray] = []
    dir_names:      List[str]        = []

    for idx, (d, img_paths) in enumerate(dir_images.items()):
        per_image = embed_images(img_paths, model, transform, batch_size, device)
        target, flat_emb = make_target_and_embedding(per_image, shuffle=shuffle)

        all_embeddings.append(flat_emb)
        all_targets.append(target[np.newaxis, :])
        dir_names.append(d.name)

        target_preview = target[:6].tolist()
        if len(target) > 6:
            target_preview = str(target_preview)[:-1] + ", ...]"
        print(f"[INFO] [{idx+1:>3}/{len(dir_images)}]  {d.name}  "
              f"→  emb {flat_emb.shape}   target {target_preview}")

    embeddings_arr = np.vstack(all_embeddings)
    targets_arr    = np.vstack(all_targets)

    print(f"\n[INFO] Embeddings        : {embeddings_arr.shape}")
    print(f"[INFO] Targets           : {targets_arr.shape}")

    if output_path:
        np.save(output_path, embeddings_arr)
        target_path = output_path.replace(".npy", "_targets.npy")
        np.save(target_path, targets_arr)
        print(f"[INFO] Saved embeddings  → {output_path}")
        print(f"[INFO] Saved targets     → {target_path}")

        if save_csv:
            csv_path    = output_path.replace(".npy", ".csv")
            emb_cols    = [f"emb_{i}"    for i in range(embeddings_arr.shape[1])]
            target_cols = [f"target_{i}" for i in range(targets_arr.shape[1])]
            header = "directory," + ",".join(target_cols) + "," + ",".join(emb_cols)
            rows = [
                f"{name},{','.join(map(str, tgt))},{','.join(map(str, emb))}"
                for name, tgt, emb in zip(dir_names, targets_arr, embeddings_arr)
            ]
            Path(csv_path).write_text(header + "\n" + "\n".join(rows))
            print(f"[INFO] Saved csv         → {csv_path}")

    return embeddings_arr, targets_arr, dir_names


# ─────────────────────────────────────────────────────────────
#  MODE 2 — ROTATE & COMPARE
# ─────────────────────────────────────────────────────────────

def run_rotation_comparison(
    image_dirs: List[str],
    num_dirs: int,
    angle_range: Tuple[float, float] = (0.0, 360.0),
    fixed_angle: Optional[float] = None,
    embedding_size: int = 512,
    batch_size: int = 16,
    image_size: Optional[Tuple[int, int]] = None,
    normalize: bool = True,
    output_path: Optional[str] = None,
    device: Optional[str] = None,
    kernel_size: int = 7,
    orientations: int = 8,
    scales: int = 4,
) -> List[Dict]:
    """
    Per-image rotation comparison using fixed Gabor embeddings.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device            : {device}")
    print(f"[INFO] Mode              : rotation comparison (per-image)")

    dirs = resolve_image_dirs(image_dirs)
    if not dirs:
        raise FileNotFoundError(f"No valid directories found for: {image_dirs}")

    valid_dirs = [d for d in dirs if sorted(d.glob("*.png"))]
    if not valid_dirs:
        raise FileNotFoundError("No PNG files found in any resolved directory.")

    if num_dirs > len(valid_dirs):
        print(f"[WARN] Requested {num_dirs} dirs but only {len(valid_dirs)} available.")
        num_dirs = len(valid_dirs)

    selected_dirs = random.sample(valid_dirs, num_dirs)
    angle_info = (f"{angle_range[0]}°–{angle_range[1]}°"
                  if fixed_angle is None else f"{fixed_angle}° (fixed)")
    print(f"[INFO] Directories       : {len(valid_dirs)} found, {num_dirs} selected")
    print(f"[INFO] Angle             : {angle_info}")
    print(f"[INFO] Embedding dim     : {embedding_size}")
    print(f"[INFO] Gabor orientations: {orientations}")
    print(f"[INFO] Gabor scales      : {scales}")
    print(f"[INFO] Gabor kernel size : {kernel_size}")
    print()

    model         = build_model(embedding_size, normalize, device, kernel_size, orientations, scales)
    transform_base = build_transform(image_size, rotation_angle=None)
    print()

    all_results, table_rows = [], []

    for d in selected_dirs:
        img_paths = sorted(d.glob("*.png"))
        angle = fixed_angle if fixed_angle is not None else round(
            random.uniform(angle_range[0], angle_range[1]), 2
        )
        transform_rot = build_transform(image_size, rotation_angle=angle)

        embs_before = embed_images(img_paths, model, transform_base, batch_size, device)
        embs_after  = embed_images(img_paths, model, transform_rot,  batch_size, device)

        try:
            short_dir = str(d.relative_to(d.parents[1]))
        except ValueError:
            short_dir = str(d)

        for img_path, emb_before, emb_after in zip(img_paths, embs_before, embs_after):
            cos_sim = cosine_similarity(emb_before, emb_after)
            l2_dist = l2_distance(emb_before, emb_after)
            all_results.append({
                "directory": str(d), "image": img_path.name, "angle": angle,
                "cosine_similarity": cos_sim, "l2_distance": l2_dist,
                "emb_before": emb_before, "emb_after": emb_after,
            })
            table_rows.append((short_dir, img_path.name, angle, cos_sim, l2_dist))

    print_table(table_rows, path_col_width=55)

    cos_vals = [r["cosine_similarity"] for r in all_results]
    l2_vals  = [r["l2_distance"]       for r in all_results]
    print(f"\n  Total images compared      : {len(all_results)}")
    print(f"  {'Mean cosine similarity':<30}: {np.mean(cos_vals):.4f}  (std {np.std(cos_vals):.4f})")
    print(f"  {'Mean L2 distance':<30}: {np.mean(l2_vals):.4f}  (std {np.std(l2_vals):.4f})")
    print(f"  {'Min / Max cosine sim':<30}: {min(cos_vals):.4f} / {max(cos_vals):.4f}")
    print(f"  {'Min / Max L2 distance':<30}: {min(l2_vals):.4f} / {max(l2_vals):.4f}")

    if output_path:
        summary = np.array(
            [(r["directory"], r["image"], r["angle"],
              r["cosine_similarity"], r["l2_distance"])
             for r in all_results],
            dtype=[("directory", "U256"), ("image", "U256"),
                   ("angle", "f4"), ("cosine_similarity", "f4"), ("l2_distance", "f4")],
        )
        np.save(output_path, summary)
        print(f"\n[INFO] Saved results → {output_path}")

    return all_results


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gabor filter bank embedding extractor — zero learnable parameters.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--image_dirs", required=True, nargs="+", metavar="DIR_OR_GLOB")
    shared.add_argument("--embedding_size", type=int, default=512,
                        help="Output embedding dimension (default: 512)")
    shared.add_argument("--batch_size",     type=int, default=16)
    shared.add_argument("--image_size",     type=int, nargs=2, default=None, metavar=("H", "W"))
    shared.add_argument("--no_normalize",   action="store_true")
    shared.add_argument("--output",         default=None)
    shared.add_argument("--device",         default=None)
    # Gabor filter bank controls
    shared.add_argument("--kernel_size",  type=int, default=7,
                        help="Gabor kernel spatial size, must be odd (default: 7)")
    shared.add_argument("--orientations", type=int, default=8,
                        help="Number of filter orientations in [0,π) (default: 8)")
    shared.add_argument("--scales",       type=int, default=4,
                        help="Number of frequency scales (default: 4)")

    # embed
    p_embed = sub.add_parser("embed", parents=[shared],
                              help="Extract concatenated per-directory embeddings.",
                              formatter_class=argparse.RawDescriptionHelpFormatter)
    p_embed.add_argument("--num_dirs",  type=int, default=None,
                         help="Randomly sample this many directories. Default: use all.")
    p_embed.add_argument("--shuffle",  action="store_true",
                         help="Randomly permute images before concatenating.")
    p_embed.add_argument("--save_csv", action="store_true")

    # rotate
    p_rot = sub.add_parser("rotate", parents=[shared],
                            help="Compare per-image embeddings before and after rotation.",
                            formatter_class=argparse.RawDescriptionHelpFormatter)
    p_rot.add_argument("--num_dirs",    type=int, required=True)
    p_rot.add_argument("--angle_range", type=float, nargs=2, default=[0.0, 360.0],
                       metavar=("MIN", "MAX"))
    p_rot.add_argument("--fixed_angle", type=float, default=None)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()

    common = dict(
        image_dirs    = args.image_dirs,
        embedding_size = args.embedding_size,
        batch_size    = args.batch_size,
        image_size    = tuple(args.image_size) if args.image_size else None,
        normalize     = not args.no_normalize,
        output_path   = args.output,
        device        = args.device,
        kernel_size   = args.kernel_size,
        orientations  = args.orientations,
        scales        = args.scales,
    )

    if args.mode == "embed":
        run_embed(**common, num_dirs=args.num_dirs, shuffle=args.shuffle, save_csv=args.save_csv)

    elif args.mode == "rotate":
        run_rotation_comparison(
            **common,
            num_dirs    = args.num_dirs,
            angle_range = tuple(args.angle_range),
            fixed_angle = args.fixed_angle,
        )