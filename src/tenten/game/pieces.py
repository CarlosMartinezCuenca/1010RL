"""The 19 pieces of 1010! as 0/1 arrays (all rotations included)."""

import numpy as np

PIECES: dict[str, np.ndarray] = {
    "dot": np.array([[1]], dtype=np.uint8),
    "domino_h": np.array([[1, 1]], dtype=np.uint8),
    "domino_v": np.array([[1], [1]], dtype=np.uint8),
    "I3_h": np.array([[1, 1, 1]], dtype=np.uint8),
    "I3_v": np.array([[1], [1], [1]], dtype=np.uint8),
    "I4_h": np.ones((1, 4), dtype=np.uint8),
    "I4_v": np.ones((4, 1), dtype=np.uint8),
    "I5_h": np.ones((1, 5), dtype=np.uint8),
    "I5_v": np.ones((5, 1), dtype=np.uint8),
    "square2": np.ones((2, 2), dtype=np.uint8),
    "square3": np.ones((3, 3), dtype=np.uint8),
    "small_L": np.array([[1, 0], [1, 1]], dtype=np.uint8),
    "small_L_b": np.array([[1, 1], [1, 0]], dtype=np.uint8),
    "small_L_c": np.array([[1, 1], [0, 1]], dtype=np.uint8),
    "small_L_d": np.array([[0, 1], [1, 1]], dtype=np.uint8),
    "big_L": np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=np.uint8),
    "big_L_b": np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=np.uint8),
    "big_L_c": np.array([[1, 1, 1], [0, 0, 1], [0, 0, 1]], dtype=np.uint8),
    "big_L_d": np.array([[0, 0, 1], [0, 0, 1], [1, 1, 1]], dtype=np.uint8),
}
