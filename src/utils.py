import numpy as np

def to_bin_array(encoded_caracter):
    bin_array = np.zeros((7, 5), dtype=int)
    for row in range(0, 7):
        current_row = encoded_caracter[row]
        for col in range(0, 5):
            bin_array[row][4-col] = current_row & 1
            current_row >>= 1
    return bin_array  

def font_dataset_to_matrix(font_data: np.ndarray) -> np.ndarray:
    dataset = []
    for encoded_char in font_data:
        binary_char = to_bin_array(encoded_char)
        dataset.append(binary_char.flatten())
    return np.array(dataset)

def add_salt_and_pepper_noise(x: np.ndarray, noise_level: float) -> np.ndarray:
    """
    Applies salt and pepper noise to a 1D binary numpy array.
    Flips random elements (0→1, 1→0) based on given noise_level.
    """
    if x.ndim != 1:
        raise ValueError("Input must be a 1D numpy array")

    noisy = x.copy()
    num_noisy = int(noise_level * x.size)

    if num_noisy == 0:
        return noisy

    indices = np.random.choice(x.size, num_noisy, replace=False)
    noisy[indices] = 1 - noisy[indices]
    return noisy
