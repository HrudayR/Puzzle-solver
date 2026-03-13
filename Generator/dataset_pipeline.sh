#!/bin/bash
set -e

# Derive paths from the script's own location (works from any machine)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATASET_DIR="$TARGET_DIR/Dataset/ILSVRC_train"
GENERATOR_DIR="$TARGET_DIR/Generator"
OUTPUT_DIR="$TARGET_DIR/Dataset"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <resize|generate|all>"
    exit 1
fi

ACTION=$1

resize_dataset() {
    ( # Run in a subshell so we don't 'cd' the main script
        echo "Starting resize operation..."
        mkdir -p "$OUTPUT_DIR/cropped"
        cd "$GENERATOR_DIR" || exit 1
        python generate_dataset.py resize "$DATASET_DIR" "$OUTPUT_DIR/cropped"
        echo "✅ Resize completed."
    )
}

generate_datasets() {
    ( # Run in a subshell
        cd "$GENERATOR_DIR" || exit 1
        
        echo "Generating 'shattered' style dataset..."
        mkdir -p "$OUTPUT_DIR/train_set_shattered"
        python generate_dataset.py generate "$OUTPUT_DIR/cropped" "$OUTPUT_DIR/train_set_shattered" --num-pieces 20 --style shattered

        echo "Generating 'curved' style dataset..."
        mkdir -p "$OUTPUT_DIR/train_set_curved"
        python generate_dataset.py generate "$OUTPUT_DIR/cropped" "$OUTPUT_DIR/train_set_curved" --num-pieces 20 --style curved

        echo "Generating 'square' style dataset..."
        mkdir -p "$OUTPUT_DIR/train_set_square"
        python generate_dataset.py generate "$OUTPUT_DIR/cropped" "$OUTPUT_DIR/train_set_square" --num-pieces 20 --style square

        echo "✅ Dataset generation completed."
    )
}

case "$ACTION" in
    resize)   resize_dataset ;;
    generate) generate_datasets ;;
    all)
        resize_dataset
        generate_datasets
        ;;
    *)
        echo "Invalid argument: $ACTION"
        exit 1
        ;;
esac