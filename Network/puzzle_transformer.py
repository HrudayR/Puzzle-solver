import torch
import torch.nn as nn
import numpy as np

# ── Embedding mode ────────────────────────────────────────────
# "stored" : load pre-computed .npy files (fast, fixed shuffles)
# "live"   : generate embeddings on the fly via PuzzleDataset
EMBEDDING_MODE = "live"

# ── Shared config ─────────────────────────────────────────────
EMBEDDING_SIZE   = 128
NUMBER_OF_PIECES = 20
BATCH_SIZE       = 8
PIECE_DIM        = 3 * EMBEDDING_SIZE

# ── Stored mode paths ─────────────────────────────────────────
STORED_X_PATH = "/home/hruday/studies/computer_vision/puzzle_solver/Puzzle-solver/Dataset/paired_embeddings_curved_128.npy"
STORED_Y_PATH = "/home/hruday/studies/computer_vision/puzzle_solver/Puzzle-solver/Dataset/targets_one_hot_curved_128.npy"

# ── Live mode config ──────────────────────────────────────────
DATASET_ROOT     = "/home/hruday/studies/computer_vision/puzzle_solver/Puzzle-solver/Dataset/train_set_curved"
GLOB_PATTERN     = "*/pieces/piece_*.png"
K_CLUSTERS       = 8
ENCODER_CHECKPOINT = None  # set to a .pt path once the encoder has been trained

def sinkhorn(log_alpha, n_iters=20):
    """Normalize a matrix to be doubly stochastic via Sinkhorn iterations (in log space)."""
    for _ in range(n_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)  # row norm
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)  # col norm
    return torch.exp(log_alpha)

def sinkhorn_loss(logits, target, n_iters=20, eps=1e-8):
    """
    Sinkhorn divergence between predicted soft permutation and one-hot target.
    logits: (B, N, N) raw scores
    target: (B, N, N) one-hot permutation matrix
    """
    # Convert logits to soft permutation matrix via Sinkhorn
    soft_perm = sinkhorn(logits, n_iters=n_iters)             # (B, N, N) doubly stochastic
    
    # Cross entropy between soft permutation and target
    loss = -(target * torch.log(soft_perm + eps)).sum(dim=(-1, -2)).mean()
    return loss


def augment_batch(x, y, k=4):
    """
    Given a batch of puzzles, create k random reshuffles of each.
    x: (B, N, D)
    y: (B, N, N)  one-hot permutation matrices
    Returns (B*k, N, D) and (B*k, N, N)
    """
    B, N, D = x.shape
    xs, ys = [], []
    for _ in range(k):
        perm = torch.stack([torch.randperm(N) for _ in range(B)])  # (B, N)
        x_shuf = torch.stack([x[b][perm[b]] for b in range(B)])    # (B, N, D)
        # Build new one-hot: y_shuf[b, s, orig] = 1 where orig = perm[b][s]
        y_shuf = torch.zeros_like(y)
        for b in range(B):
            for s in range(N):
                orig = perm[b, s].item()
                y_shuf[b, s, orig] = 1
        xs.append(x_shuf)
        ys.append(y_shuf)
    return torch.cat(xs), torch.cat(ys)

class PuzzleTransformer(nn.Module):
    def __init__(self, piece_dim=384, num_pieces=20, 
                 d_model=256, nhead=8, num_layers=4):
        super().__init__()
        self.num_pieces = num_pieces

        # Project each piece to d_model
        self.input_proj = nn.Sequential(
            nn.Linear(piece_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # Transformer encoder — permutation invariant, 
        # reasons about all pieces together
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=256, dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Output head
        self.fc = nn.Linear(d_model, num_pieces)

    def forward(self, x):
        # x: (B, N, piece_dim)
        x = self.input_proj(x)           # (B, N, d_model)
        x = self.transformer(x)          # (B, N, d_model)
        logits = self.fc(x)              # (B, N, N)
        return logits



if __name__ == "__main__":
    from torch.utils.data import DataLoader, TensorDataset, random_split

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Build DataLoaders ─────────────────────────────────────
    if EMBEDDING_MODE == "stored":
        x = torch.tensor(np.load(STORED_X_PATH), dtype=torch.float32)
        y = torch.tensor(np.load(STORED_Y_PATH), dtype=torch.float32)
        print(f"Loaded stored embeddings: x={x.shape}  y={y.shape}")
        print(f"x  mean={x.mean():.4f}  std={x.std():.4f}")

        dataset = TensorDataset(x, y)
        n_train = int(0.8 * len(dataset))
        train_ds, test_ds = random_split(dataset, [n_train, len(dataset) - n_train])
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    else:  # "live"
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from Embeddings.puzzle_embedding import PuzzleDataset, collate_fn
        dataset = PuzzleDataset(
            root            = DATASET_ROOT,
            glob_pattern    = GLOB_PATTERN,
            embed_dim       = EMBEDDING_SIZE,
            device          = device,
            k_clusters      = K_CLUSTERS,
            checkpoint_path = ENCODER_CHECKPOINT,
        )
        n_train = int(0.8 * len(dataset))
        train_ds, test_ds = random_split(dataset, [n_train, len(dataset) - n_train])
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
        test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    print(f"Mode: {EMBEDDING_MODE} | Train: {len(train_ds)} | Test: {len(test_ds)}")

    # ── Model & optimiser ─────────────────────────────────────
    model = PuzzleTransformer(piece_dim=PIECE_DIM, num_pieces=NUMBER_OF_PIECES).to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # ── Training loop ─────────────────────────────────────────
    for epoch in range(1000):
        model.train()
        epoch_loss, num_batches = 0.0, 0

        n_batches = len(train_loader)
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device) + torch.randn_like(x_batch.to(device)) * 0.01
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = sinkhorn_loss(logits, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            if num_batches % 10 == 0 or num_batches == n_batches:
                print(f"  batch {num_batches}/{n_batches} | loss {loss.item():.4f}")

        # ── Eval ─────────────────────────────────────────────
        model.eval()
        test_loss_total = 0.0
        sample_x, sample_y = None, None

        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                test_logits = model(x_batch)
                test_loss_total += sinkhorn_loss(test_logits, y_batch).item()
                if sample_x is None:
                    sample_x, sample_y = x_batch, y_batch  # keep first batch for inspection

            avg_test_loss  = test_loss_total / len(test_loader)
            avg_train_loss = epoch_loss / num_batches

            soft_predictions = sinkhorn(model(sample_x), n_iters=20)
            pred_matrix = soft_predictions[0]
            true_matrix = sample_y[0]
            pred_indices = pred_matrix.argmax(dim=-1)
            true_indices = true_matrix.argmax(dim=-1)
            confidences  = pred_matrix.max(dim=-1).values

            matches = (pred_indices == true_indices).sum().item()

        print(f"\n{'='*60}")
        print(f"  Epoch {epoch+1}/1000  |  Train Loss: {avg_train_loss:.4f}  |  Test Loss: {avg_test_loss:.4f}")
        print(f"  Correct: {matches}/{NUMBER_OF_PIECES}  |  Confidence (first 5): {[f'{c:.2f}' for c in confidences[:5].tolist()]}")
        print(f"  Target:    {true_indices.tolist()}")
        print(f"  Predicted: {pred_indices.tolist()}")
        print(f"{'='*60}\n")

    # ── Final inference ───────────────────────────────────────
    model.eval()
    with torch.no_grad():
        all_preds = []
        for x_batch, _ in test_loader:
            soft_perm = sinkhorn(model(x_batch.to(device)))
            all_preds.append(soft_perm.argmax(dim=-1))
        pred_permutation = torch.cat(all_preds)
        print(f"\nPredicted permutation shape: {pred_permutation.shape}")
        print(f"Predicted permutation:\n{pred_permutation}")