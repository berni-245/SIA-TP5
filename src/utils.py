import numpy as np

def to_bin_array(char_pattern: list[int]) -> np.ndarray:
    """
    Convierte una lista de 7 enteros (un carácter) a un array binario 7x5.
    """
    bin_array = np.zeros((7, 5), dtype=np.float64)
    for row_idx, val in enumerate(char_pattern):
        for col_idx in range(5):
            if val & (1 << (4 - col_idx)):
                bin_array[row_idx, col_idx] = 1.0
    return bin_array

def font_dataset_to_matrix(font_array: list[list[int]]) -> np.ndarray:
    """
    Convierte la lista Font3[32][7] a una matriz de shape (32, 35).
    """
    result = []
    for char in font_array:
        bin_matrix = to_bin_array(char)
        result.append(bin_matrix.flatten())  # 7x5 => 35
    return np.array(result)
