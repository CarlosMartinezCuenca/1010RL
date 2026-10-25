"""Tests for the linear probe of the n-tuple value function."""

import numpy as np

from tenten.agents.probing import FEATURES, board_features, fit_probe
from tenten.game.env import TenTenEnv


def test_features_of_empty_board():
    env = TenTenEnv()
    f = dict(zip(FEATURES, board_features(env.engine, np.zeros(100, dtype=bool))))
    assert f["empty_cells"] == 1.0
    assert f["blocked_pieces"] == 0.0
    assert f["isolated_holes"] == 0
    assert f["almost_full_lines"] == 0


def test_isolated_hole_and_almost_full_line():
    env = TenTenEnv()
    board = np.zeros((10, 10), dtype=bool)
    board[0, :] = True
    board[0, 5] = False  # row 0: 9 filled cells and one hole...
    board[1, 5] = True  # ...walled in from below (the edge is above)
    board[0, 4] = board[0, 6] = True
    f = dict(zip(FEATURES, board_features(env.engine, board.ravel())))
    assert f["isolated_holes"] == 1
    assert f["almost_full_lines"] == 1


def test_probe_finds_a_linear_relation():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 3))
    v = 2 * x[:, 0] - x[:, 1] + 0.01 * rng.normal(size=500)
    probe = fit_probe(x, v)
    assert probe["r2"] > 0.99
    assert probe["correlations"][0] > 0.8 and probe["correlations"][1] < -0.3
