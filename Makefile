TARGET_DIR = /home/hruday/studies/computer_vision/puzzle_solver/Puzzle-solver
DATASET_DIR = /home/hruday/studies/computer_vision/puzzle_solver/Puzzle-solver/Dataset


.PHONY: all single clean

all: single

single:
	cd $(TARGET_DIR)/Generator && ./generate_single.sh 50 shattered

resize:
	cd $(TARGET_DIR)/Generator && ./dataset_pipeline.sh resize


generate_dataset:
	cd $(TARGET_DIR)/Generator && ./dataset_pipeline.sh generate

clean:
	rm -rf $(TARGET_DIR)/Generator/shattered_*
	rm -rf $(TARGET_DIR)/Generator/curved_*
