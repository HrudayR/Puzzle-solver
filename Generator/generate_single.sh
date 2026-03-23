#!/bin/bash

PUZZLE_DIR="/home/hruday/studies/computer_vision/puzzle_solver/Puzzle-solver" # <Path to your puzzle solver directory>
IMAGE_PATH="$PUZZLE_DIR/Generator/tard-58b8d17f3df78c353c22729d.jpg" # <Path to your input image>
# N_PUZZLES=${1:-50} # Number of puzzles to generate (default: 50)
# STYLE=${2:-shattered} # Style of the puzzles (default: shattered) [options: shattered, curved]

N_PUZZLES=20 # Number of puzzles to generate (default: 50)
STYLE="shattered" # Style of the puzzles (default: shattered) [options: shattered, curved]

cd "$PUZZLE_DIR/Generator"
python puzzle_generator.py "$IMAGE_PATH" -n "$N_PUZZLES" -s "$STYLE"
echo "Generated $N_PUZZLES puzzles with style '$STYLE'"

