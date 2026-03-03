import os
from pathlib import Path

def check_dataset_consistency(root_dir):
    root = Path(root_dir)
    # Get all 'pieces' directories
    pieces_dirs = list(root.glob("**/pieces"))
    
    if not pieces_dirs:
        print(f"❌ No 'pieces' directories found in {root_dir}")
        return

    # Track findings
    stats = {}
    mismatched = []
    non_png_files = []

    print(f"🔍 Scanning {len(pieces_dirs)} directories...")

    for p_dir in pieces_dirs:
        # List all files in the current pieces folder
        all_files = [f for f in p_dir.iterdir() if f.is_file()]
        png_files = [f for f in all_files if f.suffix.lower() == '.png']
        
        count = len(png_files)
        stats[str(p_dir)] = count

        # Check for non-PNG intruders
        if len(all_files) != len(png_files):
            non_png_files.append((str(p_dir), len(all_files) - len(png_files)))

    # Determine the "expected" count (using the most common count found)
    counts = list(stats.values())
    expected_count = max(set(counts), key=counts.count)

    # Find directories that don't match the majority
    for path, count in stats.items():
        if count != expected_count:
            mismatched.append((path, count))

    # --- Report Results ---
    print("-" * 30)
    if not mismatched and not non_png_files:
        print(f"✅ Success! All directories have {expected_count} PNG files.")
    else:
        if mismatched:
            print(f"⚠️ Found {len(mismatched)} directories with incorrect counts (Expected {expected_count}):")
            for path, count in mismatched:
                print(f"  - {path}: {count} files")
        
        if non_png_files:
            print(f"\n🚫 Found non-PNG files in these directories:")
            for path, extra in non_png_files:
                print(f"  - {path}: {extra} non-PNG file(s) detected")

if __name__ == "__main__":
    check_dataset_consistency("../Dataset/train_set_curved/")
    check_dataset_consistency("../Dataset/train_set_shattered/")