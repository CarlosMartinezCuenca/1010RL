"""
Probing: does the learned value V(board) track human-understandable board features?

We collect boards from the agent's own games, compute a few interpretable features for
each one and measure how well they explain V (correlation and a linear fit).
"""

import numpy as np

FEATURES = [
    "empty_cells",
    "mobility",
    "blocked_pieces",
    "isolated_holes",
    "almost_full_lines",
]


def board_features(engine, board):
    """Interpretable features of one board, in the order of FEATURES.

    - empty_cells:       fraction of empty cells
    - mobility:          mean over the pieces of log(1 + number of places where it fits)
                         (the same quantity the heuristic baseline uses)
    - blocked_pieces:    fraction of pieces that fit nowhere
    - isolated_holes:    empty cells whose 4 neighbours are all filled (or the edge)
    - almost_full_lines: rows and columns with at most 2 empty cells
    """
    size = engine.size
    grid = board.reshape(size, size)

    places = np.array([engine.valid(board, k).sum() for k in range(len(engine.pieces))])

    walled = np.pad(grid, 1, constant_values=True)  # the edge counts as a wall
    neighbours_filled = (
        walled[:-2, 1:-1] & walled[2:, 1:-1] & walled[1:-1, :-2] & walled[1:-1, 2:]
    )

    return np.array(
        [
            1 - board.mean(),
            np.log1p(places).mean(),
            (places == 0).mean(),
            (~grid & neighbours_filled).sum(),
            (grid.sum(axis=1) >= size - 2).sum() + (grid.sum(axis=0) >= size - 2).sum(),
        ],
        dtype=float,
    )


def fit_probe(features, values):
    """How much of V the features explain.

    features: (N, F) array, values: (N,) array of V.
    Returns the correlation of V with each feature, the standardised coefficients of a
    linear fit V ~ features (comparable between features) and its R² (0 = nothing, 1 = all).
    """

    def standardise(x):
        std = x.std(axis=0)
        return (x - x.mean(axis=0)) / np.where(std > 0, std, 1)

    x, y = standardise(features), standardise(values)
    correlations = (x * y[:, None]).mean(axis=0)  # Pearson r (0 for a constant feature)
    coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
    r2 = 1 - ((y - x @ coefficients) ** 2).mean()  # y has variance 1
    return {"correlations": correlations, "coefficients": coefficients, "r2": float(r2)}
