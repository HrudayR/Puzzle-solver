#!/usr/bin/env python3
"""
Rotate puzzle piece images by random angles and save to a parallel directory structure.
Edit the global variables below, then run: python rotate_pieces.py
"""

import os
import random
from pathlib import Path
from PIL import Image

# ── Configuration ─────────────────────────────────────────────────────────────

INPUT_DIR     = "/home/hruday/studies/computer_vision/puzzle_solver/Puzzle-solver/Dataset/train_set_curved"
OUTPUT_SUFFIX = "_rotated"   # appended to each image-folder name
ANGLE_MIN     = 0.0          # minimum rotation in degrees
ANGLE_MAX     = 360.0        # maximum rotation in degrees
SEED          = 42         # set to an int for reproducible angles, e.g. 42

# ──────────────────────────────────────────────────────────────────────────────


def rotate_and_save(input_path: str, output_path: str, angle: float):
    img = Image.open(input_path).convert("RGBA")
    rotated = img.rotate(-angle, expand=True, resample=Image.BICUBIC)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rotated.save(output_path, "PNG")


def main():
    if SEED is not None:
        random.seed(SEED)

    input_root = Path(INPUT_DIR)
    png_files = sorted(input_root.rglob("*/pieces/*.png"))

    if not png_files:
        print(f"No PNG files found under {input_root}")
        return

    print(f"Found {len(png_files)} PNG file(s). Processing...\n")

    for png_path in png_files:
        relative_to_root = png_path.relative_to(input_root)
        image_folder = relative_to_root.parts[0]           # e.g. ILSVRC2017_test_00000001
        rest_of_path = Path(*relative_to_root.parts[1:])   # e.g. pieces/piece_002.png

        output_path = input_root / (image_folder + OUTPUT_SUFFIX) / rest_of_path
        angle = random.uniform(ANGLE_MIN, ANGLE_MAX)

        rotate_and_save(str(png_path), str(output_path), angle)
        print(f"  [{angle:6.1f}°]  {png_path.name}  ->  {output_path}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()