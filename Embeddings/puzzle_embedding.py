import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset

from Embeddings.conv_learnable_embeddings import PieceEncoder, embed_images
from Embeddings.fourier_descriptor import get_fourier_descriptors
from Embeddings.color_embeddings import get_color_embedding


def collect_puzzles(root, glob_pattern):
    """Return list of (puzzle_dir, [piece_paths]) for each puzzle directory."""
    root = Path(root)
    all_paths = sorted(root.rglob(glob_pattern))
    puzzles = {}
    for p in all_paths:
        puzzles.setdefault(p.parent, []).append(p)
    return [(puzzle_dir, sorted(paths)) for puzzle_dir, paths in sorted(puzzles.items())]


def shuffle_and_pair(conv, fourier, color):
    """
    Shuffle pieces and return packed embeddings, permutation target, and one-hot target.

    Args:
        conv:    (N, D) numpy array of learnable conv embeddings
        fourier: (N, D) numpy array of Fourier descriptors
        color:   (N, D) numpy array of color embeddings

    Returns:
        packed:  (N, 3D) tensor — shuffled, all three embeddings concatenated per piece
        target:  (N,)   tensor — original indices (permutation)
        one_hot: (N, N) tensor — one-hot encoding of target
    """
    N = conv.shape[0]
    perm = np.random.permutation(N)

    packed = np.concatenate([conv[perm], fourier[perm], color[perm]], axis=1)  # (N, 3D)

    target = torch.from_numpy(perm).long()
    one_hot = torch.zeros(N, N)
    one_hot[torch.arange(N), target] = 1.0

    return torch.from_numpy(packed).float(), target, one_hot


class PuzzleDataset(Dataset):
    def __init__(self, root, glob_pattern, embed_dim, device, k_clusters=8, checkpoint_path=None):
        self.puzzles = collect_puzzles(root, glob_pattern)
        self.encoder = PieceEncoder(embed_dim).to(device).eval()
        if checkpoint_path is not None:
            self.encoder.load_state_dict(torch.load(checkpoint_path, map_location=device))
        self.embed_dim = embed_dim
        self.device = device
        self.k_clusters = k_clusters

    def __len__(self):
        return len(self.puzzles)

    def __getitem__(self, idx):
        _, piece_paths = self.puzzles[idx]

        conv_emb = embed_images(piece_paths, self.encoder, self.device)   # (N, D) numpy

        fourier_emb = np.stack([
            get_fourier_descriptors(str(p), self.embed_dim) for p in piece_paths
        ])                                                                  # (N, D) numpy

        color_emb = np.stack([
            get_color_embedding(str(p), self.embed_dim, self.k_clusters) for p in piece_paths
        ])                                                                  # (N, D) numpy

        return conv_emb, fourier_emb, color_emb


def collate_fn(batch):
    # batch = list of (conv_emb, fourier_emb, color_emb) tuples
    packed_list, one_hot_list = [], []
    for conv, fourier, color in batch:
        packed, _, one_hot = shuffle_and_pair(conv, fourier, color)
        packed_list.append(packed)
        one_hot_list.append(one_hot)
    return torch.stack(packed_list), torch.stack(one_hot_list)
