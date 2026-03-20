import torch
import torch.nn as nn
import numpy as np

EMBEDDING_SIZE = 64
NUMBER_OF_PIECES = 20

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


class PuzzleNet(nn.Module):
    def __init__(self, d=2560, pair_dim=1, num_pieces=20):
        super(PuzzleNet, self).__init__()
        
        self.num_pieces = num_pieces
        
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d * pair_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_pieces * num_pieces)
        )
    
    def forward(self, x):
        logits = self.network(x)
        return logits.view(-1, self.num_pieces, self.num_pieces)


class PuzzleNetWithAttention(nn.Module):
    def __init__(self, d=2560, pair_dim=1, num_pieces=20, embed_dim=256, num_heads=4):
        super(PuzzleNetWithAttention, self).__init__()
        
        self.num_pieces = num_pieces

        # First stage: projection and attention
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(d * pair_dim, embed_dim)
        self.relu1 = nn.ReLU()
        
        # Multi-head self-attention block
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        
        # Second stage: feed-forward layers
        self.fc2 = nn.Linear(embed_dim, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        
        self.output = nn.Linear(128, num_pieces * num_pieces)

    def forward(self, x):
        # Flatten spatial dimensions but preserve batch
        x = self.flatten(x)               # (batch, d * pair_dim)
        x = self.fc1(x)
        x = self.relu1(x)

        # Expand into a sequence for attention — treating each input as one token
        # If you have multiple pair features, replace unsqueeze with reshaping logic to (batch, seq_len, embed_dim)
        x = x.unsqueeze(1)                # (batch, seq_len=1, embed_dim)
        x, _ = self.attn(x, x, x)         # Self-attention
        x = self.dropout1(x)
        x = x.squeeze(1)                  # (batch, embed_dim)
        
        # Continue through MLP
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        logits = self.output(x)

        # Output shape reshaped for puzzle grid
        return logits.view(-1, self.num_pieces, self.num_pieces)




if __name__ == "__main__":
    x = np.load("/home/hruday/studies/computer_vision/puzzle_solver/Puzzle-solver/Dataset/curved_embeddings_20/paired_embeddings.npy")
    y = np.load("/home/hruday/studies/computer_vision/puzzle_solver/Puzzle-solver/Dataset/curved_embeddings_20/targets_one_hot.npy")

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

    # Legacy paired embeddings: two modalities × EMBEDDING_SIZE per piece → pair_dim=2
    model = PuzzleNet(
        d=NUMBER_OF_PIECES * EMBEDDING_SIZE,
        pair_dim=2,
        num_pieces=NUMBER_OF_PIECES,
    )
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1000):
        model.train()
        optimizer.zero_grad()
        logits = model(x_train)                              # (B, 24, 24)
        loss = sinkhorn_loss(logits, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            test_logits = model(x_test)
            test_loss = sinkhorn_loss(test_logits, y_test)

        print(f"Epoch {epoch+1}/1000 | Train Loss: {loss.item():.4f} | Test Loss: {test_loss.item():.4f}")

    # Inference on test set
    model.eval()
    with torch.no_grad():
        test_logits = model(x_test)
        soft_perm = sinkhorn(test_logits)                    # (B, 24, 24) doubly stochastic
        pred_permutation = soft_perm.argmax(dim=-1)          # (B, 24) hard permutation
        print(f"\nPredicted permutation:\n{pred_permutation}")
        print(f"\nPredicted permutation shape: {pred_permutation.shape}")