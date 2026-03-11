import torch
import torch.nn as nn
import numpy as np

class PuzzleNet(nn.Module):
    def __init__(self, d=1280, pair_dim=2, num_pieces=20):
        super(PuzzleNet, self).__init__()
        
        self.num_pieces = num_pieces
        
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d * pair_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_pieces * num_pieces)  # 20 * 20 = 400
        )
    
    def forward(self, x):
        logits = self.network(x)                          # (B, 400)
        return logits.view(-1, self.num_pieces, self.num_pieces)  # (B, 20, 20)


if __name__ == "__main__":
    x = np.load("/home/hruday/studies/computer_vision/puzzle_solver/Puzzle-solver/Dataset/paired_embeddings.npy")
    y = np.load("/home/hruday/studies/computer_vision/puzzle_solver/Puzzle-solver/Dataset/targets_one_hot.npy")

    # x: (B, 1280, 2), y: (B, 20, 20) one-hot
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    # Shuffle
    indices = torch.randperm(x.shape[0])
    x = x[indices]
    y = y[indices]

    model = PuzzleNet()
    print(model)
    print(f"Input shape:  {x.shape}")
    print(f"Target shape: {y.shape}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(500):
        optimizer.zero_grad()
        logits = model(x)                     # (B, 20, 20)
        # CrossEntropyLoss expects (B, C, ...) — C=num_classes is dim 2 here
        loss = criterion(logits.permute(0, 2, 1), y.permute(0, 2, 1))
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/100 | Loss: {loss.item():.4f}")

    # Inference
    with torch.no_grad():
        logits = model(x)                          # (B, 20, 20)
        pred_permutation = logits.argmax(dim=-1)   # (B, 20) — predicted piece per position
        print(f"\nPredicted permutation:\n{pred_permutation}")
        print(f"\nPredicted permutation:\n{pred_permutation.shape}")