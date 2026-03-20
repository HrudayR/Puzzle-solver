#!/usr/bin/env bash
set -euo pipefail

# Train PuzzleNet twice: frozen "ours" encoder, then frozen baseline encoder.
# Same CLI flags are forwarded to both runs (dataset, epochs, device, etc.).
#
# Usage:
#   ./Network/train_both_encoders.sh
#   ./Network/train_both_encoders.sh --dataset-path Dataset/train_set_square --num-pieces 20 --epochs 500
#
# Optional environment:
#   BASELINE_ENCODER   baseline_square (default) | baseline_curved
#   BASELINE_CHECKPOINT  path to PieceEncoder .pt (otherwise train.py defaults apply)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

BASELINE_ENCODER="${BASELINE_ENCODER:-baseline_square}"

echo "=== PuzzleNet: encoder=ours ==="
python -m Network.train --encoder ours "$@"

echo ""
echo "=== PuzzleNet: encoder=${BASELINE_ENCODER} ==="
if [[ -n "${BASELINE_CHECKPOINT:-}" ]]; then
  python -m Network.train --encoder "$BASELINE_ENCODER" --checkpoint "$BASELINE_CHECKPOINT" "$@"
else
  python -m Network.train --encoder "$BASELINE_ENCODER" "$@"
fi

echo ""
echo "Done: trained with ours, then ${BASELINE_ENCODER}."
