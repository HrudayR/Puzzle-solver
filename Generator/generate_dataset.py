import sys
import argparse
import statistics
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import io
import contextlib
import shutil

from puzzle_generator import create_shattered, create_jigsaw, create_square, preview_assembled_shattered, preview_assembled, preview_grid_shattered, preview_grid


@contextlib.contextmanager
def suppress_stdout():
    with io.StringIO() as buf, contextlib.redirect_stdout(buf):
        yield


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
STAT_CHOICES = ["median", "mean", "min", "max"]



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_image_paths(directory: Path, recursive: bool) -> list[Path]:
    glob = directory.rglob if recursive else directory.glob
    return [
        p for p in glob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def compute_target_size(image_paths: list[Path], stat: str) -> tuple[int, int]:
    """Derive target (width, height) from the chosen statistic."""
    widths, heights = [], []
    for path in image_paths:
        with Image.open(path) as img:
            w, h = img.size
            widths.append(w)
            heights.append(h)

    fn = {
        "median": lambda v: int(statistics.median(v)),
        "mean":   lambda v: int(statistics.mean(v)),
        "min":    lambda v: int(min(v)),
        "max":    lambda v: int(max(v)),
    }[stat]

    return fn(widths), fn(heights)


def resize_and_crop(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Scale to cover target dimensions, then center-crop to exact size."""
    src_w, src_h = img.size
    scale = max(target_w / src_w, target_h / src_h)
    new_w, new_h = int(src_w * scale), int(src_h * scale)

    img = img.resize((new_w, new_h), Image.LANCZOS)

    left = (new_w - target_w) // 2
    top  = (new_h - target_h) // 2
    return img.crop((left, top, left + target_w, top + target_h))


def seed_from_image(image_path: Path) -> int:
    """Stable per-image seed derived from the filename, so every image gets
    unique cut patterns but re-runs on the same file are reproducible."""
    return hash(image_path.name) & 0x7FFFFFFF


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def process(args: argparse.Namespace) -> None:
    input_path  = Path(args.input_dir)
    output_path = Path(args.output_dir)

    if not input_path.is_dir():
        print(f"Error: '{args.input_dir}' is not a valid directory.")
        sys.exit(1)

    image_paths = get_image_paths(input_path, args.recursive)
    if not image_paths:
        print(f"No supported images found in '{args.input_dir}'.")
        sys.exit(1)

    print(f"Found {len(image_paths)} image(s).")

    # Determine target size
    if args.size:
        try:
            target_w, target_h = map(int, args.size.lower().split("x"))
        except ValueError:
            print("Error: --size must be in WxH format, e.g. 640x480")
            sys.exit(1)
        print(f"Target size (manual): {target_w} x {target_h} px")
    else:
        target_w, target_h = compute_target_size(image_paths, args.stat)
        print(f"Target size ({args.stat}): {target_w} x {target_h} px")

    if args.dry_run:
        print("Dry run — no files written.")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    out_ext = f".{args.format.lower()}" if args.format else None

    ok, failed, skipped = 0, 0, 0
    for path in image_paths:
        try:
            with Image.open(path) as img:
                # Skip images already at target size (unless forced)
                if img.size == (target_w, target_h) and args.no_force:
                    print(f"  –  {path.name}  (already {target_w}×{target_h}, skipped)")
                    skipped += 1
                    continue

                if img.mode in ("RGBA", "P", "LA"):
                    img = img.convert("RGB")

                result   = resize_and_crop(img, target_w, target_h)
                suffix   = out_ext or path.suffix.lower()
                out_name = path.stem + suffix

                # Mirror subdirectory structure when recursive
                rel_dir  = path.parent.relative_to(input_path)
                dest_dir = output_path / rel_dir
                dest_dir.mkdir(parents=True, exist_ok=True)

                out_file = dest_dir / out_name
                result.save(out_file)
                print(f"  ✓  {path.name}  →  {out_file}")
                ok += 1
        except Exception as e:
            print(f"  ✗  {path.name}  —  {e}")
            failed += 1

    print(f"\nDone. {len(image_paths)} found, {ok} resized, {skipped} skipped, {failed} failed.")
    print(f"Output saved to: {output_path.resolve()}")


def generate_pieces(image_dir: Path, output_dir: Path, n_pieces: int = 20, style: str = "shattered"):
    print(f"Generating pieces for images in {image_dir}...")
    image_paths = get_image_paths(image_dir, recursive=False)
    print(f"Found {len(image_paths)} image(s).")

    with Image.open(image_paths[0]) as sample:
        img_w, img_h = sample.size

    if style == "shattered":
        for img in tqdm(image_paths, desc="Generating pieces", unit="img"):
            with suppress_stdout():
                seed = seed_from_image(img)
                pieces = create_shattered(img, num_pieces=n_pieces, output_dir=output_dir / img.stem / "pieces", seed=seed)
                (output_dir / img.stem / "previews").mkdir(parents=True, exist_ok=True)
                shutil.copy(img, output_dir / img.stem / "previews" / f"{img.stem}_shattered_preview.png")
                preview_grid_shattered(pieces=pieces, save_path=output_dir / img.stem / "previews" / f"{img.stem}_shattered_grid.png")

    elif style == "curved":
        for img in tqdm(image_paths, desc="Generating pieces", unit="img"):
            with suppress_stdout():
                seed = seed_from_image(img)
                pieces = create_jigsaw(img, num_pieces=n_pieces, shape_type="curved", output_dir=output_dir / img.stem / "pieces", seed=seed)
                (output_dir / img.stem / "previews").mkdir(parents=True, exist_ok=True)
                shutil.copy(img, output_dir / img.stem / "previews" / f"{img.stem}_shattered_preview.png")
                preview_grid(pieces=pieces, save_path=output_dir / img.stem / "previews" / f"{img.stem}_shattered_grid.png")

    elif style == "square":
        for img in tqdm(image_paths, desc="Generating pieces", unit="img"):
            with suppress_stdout():
                pieces = create_square(img, num_pieces=n_pieces, output_dir=output_dir / img.stem / "pieces")
                (output_dir / img.stem / "previews").mkdir(parents=True, exist_ok=True)
                shutil.copy(img, output_dir / img.stem / "previews" / f"{img.stem}_shattered_preview.png")
                preview_grid(pieces=pieces, save_path=output_dir / img.stem / "previews" / f"{img.stem}_shattered_grid.png")
    else:
        raise ValueError(f"Unknown style: {style}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="resize_images.py",
        description="Batch-resize images and generate puzzle pieces.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── resize ──────────────────────────────────────────────────────────────
    resize_parser = subparsers.add_parser("resize", help="Resize images to a common size.")
    resize_parser.add_argument("input_dir",  help="Directory containing source images.")
    resize_parser.add_argument("output_dir", help="Directory where resized images are saved.")

    size_group = resize_parser.add_mutually_exclusive_group()
    size_group.add_argument("--stat", choices=STAT_CHOICES, default="median", metavar="STAT",
                            help=f"Statistic for target size. Choices: {', '.join(STAT_CHOICES)}. Default: median.")
    size_group.add_argument("--size", metavar="WxH",
                            help="Fixed target size, e.g. 640x480.")

    resize_parser.add_argument("--format", metavar="EXT",
                               help="Convert output to this format (e.g. png, jpg).")
    resize_parser.add_argument("--recursive", "-r", action="store_true",
                               help="Recurse into subdirectories.")
    resize_parser.add_argument("--no-force", action="store_true",
                               help="Skip images already at the target size.")
    resize_parser.add_argument("--dry-run", "-n", action="store_true",
                               help="Preview target size without writing files.")

    # ── generate ────────────────────────────────────────────────────────────
    gen_parser = subparsers.add_parser("generate", help="Generate puzzle pieces from images.")
    gen_parser.add_argument("input_dir",  help="Directory containing source images.")
    gen_parser.add_argument("output_dir", help="Directory where pieces are saved.")
    gen_parser.add_argument("--num-pieces", type=int, default=20,
                            help="Number of pieces per image. Default: 20.")
    gen_parser.add_argument("--style", choices=["shattered", "curved", "square"], default="shattered",
                            help="Puzzle style. Default: shattered.")

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()

    if args.command == "resize":
        process(args)
    elif args.command == "generate":
        generate_pieces(
            image_dir=Path(args.input_dir),
            output_dir=Path(args.output_dir),
            n_pieces=args.num_pieces,
            style=args.style,
        )