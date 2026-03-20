import torch
import torch.nn as nn
import numpy as np

EMBEDDING_SIZE = 128
NUMBER_OF_PIECES = 20
BATCH_SIZE = 8
PIECE_DIM = 3 * EMBEDDING_SIZE

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
    x = np.load("/home/hruday/studies/computer_vision/puzzle_solver/Puzzle-solver/Dataset/paired_embeddings_curved_128.npy")
    y = np.load("/home/hruday/studies/computer_vision/puzzle_solver/Puzzle-solver/Dataset/targets_one_hot_curved_128.npy")

    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    print(x.mean(), x.std())

    # Shuffle
    indices = torch.randperm(x.shape[0])
    x = x[indices]
    y = y[indices]

    # Train/test split (80/20)
    split = int(0.8 * x.shape[0])
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"Train size: {x_train.shape[0]} | Test size: {x_test.shape[0]}")

    model = PuzzleTransformer(piece_dim=PIECE_DIM, num_pieces=NUMBER_OF_PIECES)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_train = x_train.shape[0]

    for epoch in range(1000):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        # Mini-batch training
        perm = torch.randperm(num_train)
        for i in range(0, num_train, BATCH_SIZE):
            batch_idx = perm[i:i + BATCH_SIZE]
            x_batch = x_train[batch_idx]
            y_batch = y_train[batch_idx]
            # x_batch, y_batch = augment_batch(x_train[batch_idx], y_train[batch_idx], k=4)

            optimizer.zero_grad()
            x_batch = x_batch + torch.randn_like(x_batch) * 0.01
            logits = model(x_batch)
            loss = sinkhorn_loss(logits, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_train_loss = epoch_loss / num_batches

        model.eval()
        with torch.no_grad():
            test_logits = model(x_test)
            test_loss = sinkhorn_loss(test_logits, y_test)

            soft_predictions = sinkhorn(test_logits, n_iters=20) 
    
            # 2. Pick the very first sample in the test set to inspect
            sample_idx = 0
            pred_matrix = soft_predictions[sample_idx]  # Shape: (20, 20)
            true_matrix = y_test[sample_idx]           # Shape: (20, 20)
            
            # 3. Convert matrices to "Hard" assignments (indices)
            # This tells us: "Slot 0 -> Piece X, Slot 1 -> Piece Y..."
            pred_indices = pred_matrix.argmax(dim=-1)
            true_indices = true_matrix.argmax(dim=-1)
            
            # 4. Get the confidence (probability) the model had for its choices
            confidences = pred_matrix.max(dim=-1).values

            print(f"\n--- Epoch {epoch+1} Sample Results ---")
            print(f"Target:    {true_indices.tolist()}")
            print(f"Predicted: {pred_indices.tolist()}")
            
            # Calculate matches
            matches = (pred_indices == true_indices).sum().item()
            print(f"Correct Assignments in this sample: {matches}/{NUMBER_OF_PIECES}")
            
            # Optional: Print the first 5 slot confidences
            print(f"Confidence (first 5 slots): {confidences[:5].tolist()}")
            print("--------------------------------------\n")

        print(f"Epoch {epoch+1}/1000 | Train Loss: {loss.item():.4f} | Test Loss: {test_loss.item():.4f}")

    # Inference on test set
    model.eval()
    with torch.no_grad():
        test_logits = model(x_test)
        soft_perm = sinkhorn(test_logits)                    # (B, 24, 24) doubly stochastic
        pred_permutation = soft_perm.argmax(dim=-1)          # (B, 24) hard permutation
        print(f"\nPredicted permutation:\n{pred_permutation}")
        print(f"\nPredicted permutation shape: {pred_permutation.shape}")