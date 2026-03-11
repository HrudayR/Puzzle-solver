import numpy as np
import sys
# np.set_printoptions(threshold=sys.maxsize)


OUTPUT_DIRECTORY  = "/home/hruday/studies/computer_vision/puzzle_solver/Puzzle-solver/Dataset/"
EMBEDDING_SIZE = 64


def pair_embeddings(conv_embeddings, fourier_embeddings):
    """
    Pair conv and fourier embeddings element-wise.
    
    Args:
        conv_embeddings: np.ndarray of shape (N, D)
        fourier_embeddings: np.ndarray of shape (N, D)
    
    Returns:
        np.ndarray of shape (N, D, 2) where last dim is [conv_val, fourier_val]
    """
    assert conv_embeddings.shape == fourier_embeddings.shape, \
        f"Shapes must match: {conv_embeddings.shape} vs {fourier_embeddings.shape}"
    
    packed_array = np.stack([conv_embeddings, fourier_embeddings], axis=-1)
    target_array = np.array([np.arange(20) for x in packed_array])

    return packed_array, target_array


def shuffle_arrays(packed_array, target_array):
    final_packed = []
    final_target = []
    final_target_one_hot = []

    for i in range(len(target_array)):
        indices = np.random.permutation(len(target_array[i]))
        shuffled_target = []
        shuffled_packed = []
        shuffled_target_one_hot = []

        for j in indices:
            shuffled_target.append(target_array[i][j])
            chunk = packed_array[i][j * EMBEDDING_SIZE : (j + 1) * EMBEDDING_SIZE]
            shuffled_packed.extend(chunk)
            one_hot_vector = [0] * 20
            one_hot_vector[j] = 1
            shuffled_target_one_hot.append(one_hot_vector)


        shuffled_packed = np.array(shuffled_packed)
        shuffled_target = np.array(shuffled_target)
        shuffled_target_one_hot = np.array(shuffled_target_one_hot)

        # Verification
        for rank, original_idx in enumerate(shuffled_target):
            original_chunk = packed_array[i][original_idx * EMBEDDING_SIZE : (original_idx + 1) * EMBEDDING_SIZE]
            shuffled_chunk = shuffled_packed[rank * EMBEDDING_SIZE : (rank + 1) * EMBEDDING_SIZE]
            assert np.allclose(original_chunk, shuffled_chunk), \
                f"Mismatch at rank {rank} (original index {original_idx})"

        print("Shuffle verified correctly!")
        final_packed.append(shuffled_packed)
        final_target.append(shuffled_target)
        final_target_one_hot.append(shuffled_target_one_hot)
    
    return np.array(final_packed), np.array(final_target), np.array(final_target_one_hot)


if __name__ == "__main__":
    conv_embeddings = np.load("/home/hruday/studies/computer_vision/puzzle_solver/Puzzle-solver/Dataset/convolution.npy")
    fourier_embeddings = np.load("/home/hruday/studies/computer_vision/puzzle_solver/Puzzle-solver/Dataset/fourier.npy")

    print(f"Conv Embeddings Shape:    {conv_embeddings.shape}")
    print(f"Fourier Embeddings Shape: {fourier_embeddings.shape}")

    packed, target = pair_embeddings(conv_embeddings, fourier_embeddings)
    packed, target, target_one_hot = shuffle_arrays(packed_array=packed, target_array=target)

    print(f"\nPaired Shape: {packed.shape}  (N, D, 2)")
    print(f"\nTarget Shape: {target.shape}  (N, D)")

    # # Verify alignment
    # assert np.allclose(packed[:, :, 0], conv_embeddings),    "Conv values misaligned!"
    # assert np.allclose(packed[:, :, 1], fourier_embeddings), "Fourier values misaligned!"

    
    # print(packed)
    # print(target)
    # Save
    np.save(OUTPUT_DIRECTORY + "/paired_embeddings.npy", packed)
    np.save(OUTPUT_DIRECTORY + "/targets.npy", target)
    np.save(OUTPUT_DIRECTORY + "/targets_one_hot.npy", target_one_hot)