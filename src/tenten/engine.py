import numpy as np

from tenten.pieces import PIECES


class Engine:
    def __init__(self, size: int, piece_names: list[str]):
        self.size = size
        self.n = size * size
        self.piece_names = list(piece_names)
        self.pieces = [PIECES[name] for name in piece_names]

        self.masks = []
        self.anchors = []
        for piece in self.pieces:
            h, w = piece.shape
            ms, an = [], []
            for row in range(size - h + 1):
                for col in range(size - w + 1):
                    m = np.zeros((size, size), dtype=bool)
                    m[row : row + h, col : col + w] = piece.astype(bool)
                    ms.append(m.ravel())
                    an.append(row * size + col)
            self.masks.append(np.array(ms))
            self.anchors.append(np.array(an))

    def valid(self, board: np.ndarray, k: int) -> np.ndarray:
        return ~(self.masks[k] & board).any(axis=1)

    def fits(self, board: np.ndarray, k: int) -> bool:
        return bool(self.valid(board, k).any())

    def mask_at(self, k: int, anchor: int) -> np.ndarray | None:
        idx = np.flatnonzero(self.anchors[k] == anchor)
        if idx.size == 0:
            return None
        return self.masks[k][idx[0]]

    def clear(self, board: np.ndarray) -> tuple[np.ndarray, int]:
        b = board.reshape(self.size, self.size).copy()
        full_rows = b.all(axis=1)
        full_cols = b.all(axis=0)
        b[full_rows, :] = False
        b[:, full_cols] = False
        return b.ravel(), int(full_rows.sum() + full_cols.sum())

    @staticmethod
    def score(cells: int, lines: int) -> int:
        return cells + 5 * lines * (lines + 1)
