# Heck et al. baseline encoder (`Encoder/baseline`)

- **`model.py`** — `EdgeCNN`, `PieceEncoder`, and piece loading (`load_piece`, `get_edge_strip`, …).
- **`encoder.py`** — `BaselineEncoder`: frozen `PieceEncoder` + checkpoint loading for PuzzleNet training.
- **`data.py`** — `PuzzleDataset`, `TripletDataset`, `make_datasets` for Phase-1 training.
- **`losses.py`** — triplet loss (cosine dissimilarity).
- **`train_encoder.py`** — CLI to run **Phase 1** (triplet pre-training), writes `encoder_phase1.pt` under `--checkpoint-dir`.

## Train the CNN encoder (Phase 1)

```bash
python -m Encoder.baseline.train_encoder \
  --dataset-path ./Dataset/train_set_square \
  --n-pieces 20 \
  --checkpoint-dir ./Encoder/baseline/checkpoints
```

Checkpoints default to `Encoder/baseline/checkpoints/encoder_phase1.pt`, which `Network/train.py` uses for `--encoder baseline_square` unless you pass `--checkpoint`.
