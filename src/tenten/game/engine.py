"""Game rules: piece placement, line clearing and scoring, vectorised with NumPy."""

import numpy as np

from tenten.game.pieces import PIECES


class Engine:
    """Precomputes every placement of every piece on a size x size board."""

    def __init__(self, size: int, piece_names: list[str]):
        self.size = size
        self.n = size * size
        self.piece_names = list(piece_names)
        self.pieces = [PIECES[name] for name in piece_names]

        self.masks = []  # masks[k][i]: flat board with piece k at its i-th position
        self.anchors = []  # anchors[k][i]: top-left cell of that position
        self.rows = []  # rows[k][anchor]: index i of that anchor, or -1 if it doesn't fit
        self.planes = []  # planes[k]: piece k drawn at the top-left corner (for observations)

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

            rows = np.full(self.n, -1)
            rows[an] = np.arange(len(an))
            self.rows.append(rows)

            plane = np.zeros((size, size), dtype=np.uint8)
            plane[:h, :w] = piece
            self.planes.append(plane.ravel())

        # All positions of all pieces stacked, and where each piece's block starts
        self.all_masks = np.concatenate(self.masks).astype(np.float32)
        self.starts = np.cumsum([0] + [len(m) for m in self.masks[:-1]])

    def valid(self, board: np.ndarray, k: int) -> np.ndarray:
        """Boolean per position of piece k: True if it doesn't overlap the board."""
        return ~(self.masks[k] & board).any(axis=1)

    def fits(self, board: np.ndarray, k: int) -> bool:
        """Whether piece k fits anywhere on the board."""
        return bool(self.valid(board, k).any())

    def mask_at(self, k: int, anchor: int) -> np.ndarray | None:
        """Cells of piece k placed at anchor, or None if it goes off the board."""
        row = self.rows[k][anchor]

        if row < 0:
            return None

        return self.masks[k][row]

    def clear(self, boards: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Clears full rows and columns of one board (n,) or a batch (B, n); returns boards and lines."""
        b = boards.reshape(-1, self.size, self.size)
        full_rows = b.all(axis=2, keepdims=True)  # (B, size, 1)
        full_cols = b.all(axis=1, keepdims=True)  # (B, 1, size)

        new = b & ~full_rows & ~full_cols
        lines = full_rows.sum(axis=(1, 2)) + full_cols.sum(axis=(1, 2))

        return new.reshape(boards.shape), lines.reshape(boards.shape[:-1])

    def count_places(self, boards: np.ndarray) -> np.ndarray:
        """Number of valid positions of every piece on each board: (B, n) -> (B, n_pieces)."""
        hits = boards.astype(np.float32) @ self.all_masks.T  # occupied cells under each position
        # Sum the free positions (hits == 0) within each piece's block of columns
        return np.add.reduceat((hits == 0).astype(np.int32), self.starts, axis=1)

    @staticmethod
    def score(cells: int, lines: int) -> int:
        """Points of a move: one per cell placed plus a bonus that grows with the lines cleared."""
        return cells + 5 * lines * (lines + 1)
