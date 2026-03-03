"""
Script to scan puzzle piece directories and report image dimensions.

Automatically finds all '/pieces' subdirectories under the train_set_curved dataset,
then prints the width x height for every image (PNG, JPG, JPEG, GIF, BMP, TIFF).
Includes error handling for corrupted files and provides per-directory summaries.

Usage: Set base_dir to your dataset path and run.
"""


import os
from PIL import Image

def get_image_dimensions(image_path):
    """Get width and height of an image."""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            return width, height
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
        return None, None

def scan_pieces_directory(base_path):
    """Scan all pieces directories under base_path and print image dimensions."""
    base_path = os.path.abspath(base_path)
    
    for root, dirs, files in os.walk(base_path):
        # Only process directories that end with '/pieces'
        if root.endswith('/pieces'):
            print(f"\n{'='*60}")
            print(f"Directory: {root}")
            print(f"{'='*60}")
            
            image_count = 0
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                    image_path = os.path.join(root, file)
                    width, height = get_image_dimensions(image_path)
                    if width is not None:
                        print(f"{file}: {width} x {height}")
                        image_count += 1
            
            print(f"Total images processed: {image_count}")

# Your base directory
base_dir = "../Dataset/train_set_shattered"

# Run the scanner
scan_pieces_directory(base_dir)