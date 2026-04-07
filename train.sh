#!/bin/bash

#SBATCH --job-name=puzzle_transformer
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4GB
#SBATCH --gpus-per-task=1
#SBATCH --output=/scratch/${USER}/logs/train_%j.out
#SBATCH --error=/scratch/${USER}/logs/train_%j.err

# ── Paths (edit these) ────────────────────────────────────────
REPO_DIR="/scratch/${USER}/Puzzle-solver"
VENV_DIR="${REPO_DIR}/puzzle_venv"

# ── Setup ─────────────────────────────────────────────────────
mkdir -p /scratch/${USER}/logs

module load 2025
module load Python/3.11
module load CUDA/12.1

# Create venv and install deps if it doesn't exist yet
if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating virtual environment..."
    python -m venv "${VENV_DIR}"
    source "${VENV_DIR}/bin/activate"
    pip install --upgrade pip
    pip install -r "${REPO_DIR}/requirements.txt"
else
    source "${VENV_DIR}/bin/activate"
fi

# ── Run ───────────────────────────────────────────────────────
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader)"

cd "${REPO_DIR}/Network"

srun python puzzle_transformer.py

echo "Job finished: $(date)"
