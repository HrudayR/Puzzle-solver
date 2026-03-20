import numpy as np
import sys

OUTPUT_DIRECTORY   = "/home/hruday/studies/computer_vision/puzzle_solver/Puzzle-solver/Dataset/"
EMBEDDING_SIZE     = 128   # conv and fourier
NUMBER_OF_PIECES   = 20
PIECE_DIM          = 3 * EMBEDDING_SIZE


def pair_embeddings(conv_embeddings, fourier_embeddings, color_embeddings):
    """
    Pair conv, fourier and color embeddings per piece.

    Args:
        conv_embeddings:    (N_puzzles, N_pieces * EMBEDDING_SIZE)
        fourier_embeddings: (N_puzzles, N_pieces * EMBEDDING_SIZE)
        color_embeddings:   (N_puzzles, N_pieces * COLOR_EMBEDDING_SIZE)

    Returns:
        packed_array: (N_puzzles, N_pieces, PIECE_DIM)
            Each piece gets [conv | fourier | color] concatenated.
        target_array: (N_puzzles, N_pieces)
    """
    N = conv_embeddings.shape[0]
    assert fourier_embeddings.shape[0] == N and color_embeddings.shape[0] == N

    conv_r    = conv_embeddings.reshape(N, NUMBER_OF_PIECES, EMBEDDING_SIZE)
    fourier_r = fourier_embeddings.reshape(N, NUMBER_OF_PIECES, EMBEDDING_SIZE)
    color_r   = color_embeddings.reshape(N, NUMBER_OF_PIECES, EMBEDDING_SIZE)

    # (N_puzzles, N_pieces, PIECE_DIM)
    packed_array = np.concatenate([conv_r, fourier_r, color_r], axis=-1)
    target_array = np.tile(np.arange(NUMBER_OF_PIECES), (N, 1))

    return packed_array, target_array


def shuffle_arrays(packed_array, target_array):
    # unchanged — works on any feature dim
    N, P, F = packed_array.shape
    final_packed        = np.zeros_like(packed_array)
    final_target        = np.zeros_like(target_array)
    final_target_onehot = np.zeros((N, P, P), dtype=np.int32)

    for i in range(N):
        perm = np.random.permutation(P)

        final_packed[i] = packed_array[i][perm]
        final_target[i] = target_array[i][perm]

        for s, orig_idx in enumerate(perm):
            final_target_onehot[i, s, orig_idx] = 1

        for s in range(P):
            assert np.allclose(final_packed[i, s], packed_array[i, perm[s]]), \
                f"Mismatch at puzzle {i}, slot {s}"

    print("Shuffle verified correctly!")
    return final_packed, final_target, final_target_onehot


if __name__ == "__main__":
    conv_embeddings    = np.load(f"{OUTPUT_DIRECTORY}/convolution_curved_{EMBEDDING_SIZE}.npy")
    fourier_embeddings = np.load(f"{OUTPUT_DIRECTORY}/fourier_curved_{EMBEDDING_SIZE}.npy")
    color_embeddings   = np.load(f"{OUTPUT_DIRECTORY}/color_curved_{EMBEDDING_SIZE}.npy")

    print(f"Conv Embeddings Shape:    {conv_embeddings.shape}")
    print(f"Fourier Embeddings Shape: {fourier_embeddings.shape}")
    print(f"Color Embeddings Shape:   {color_embeddings.shape}")

    packed, target = pair_embeddings(conv_embeddings, fourier_embeddings, color_embeddings)
    packed, target, target_one_hot = shuffle_arrays(packed_array=packed, target_array=target)

    print(f"\nPaired Shape: {packed.shape}  (N, N_pieces, {EMBEDDING_SIZE})")
    print(f"Target Shape: {target.shape}  (N, N_pieces)")

    np.save(f"{OUTPUT_DIRECTORY}/paired_embeddings_curved_{EMBEDDING_SIZE}.npy", packed)
    np.save(f"{OUTPUT_DIRECTORY}/targets_curved_{EMBEDDING_SIZE}.npy", target)
    np.save(f"{OUTPUT_DIRECTORY}/targets_one_hot_curved_{EMBEDDING_SIZE}.npy", target_one_hot)