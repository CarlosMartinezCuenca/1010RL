"""Tests for the rules engine: placements, line clearing and scoring."""

import numpy as np

from tenten.game.engine import Engine
from tenten.game.pieces import PIECES


def empty_board(size=10):
    return np.zeros(size * size, dtype=bool)


def test_there_are_19_pieces():
    assert len(PIECES) == 19


def test_number_of_positions():
    eng = Engine(10, ["dot", "square3", "I5_h"])
    assert len(eng.anchors[0]) == 100
    assert len(eng.anchors[1]) == 64
    assert len(eng.anchors[2]) == 60


def test_valid_detects_overlap():
    eng = Engine(10, ["square2"])
    board = empty_board()
    board[0] = True
    valid = eng.valid(board, 0)
    assert not valid[eng.anchors[0] == 0][0]
    assert valid[eng.anchors[0] == 2][0]


def test_mask_at():
    eng = Engine(10, ["domino_h"])
    m = eng.mask_at(0, 23)
    assert set(np.flatnonzero(m)) == {23, 24}
    assert eng.mask_at(0, 9) is None


def test_clear_one_row():
    eng = Engine(10, ["dot"])
    board = empty_board()
    board[30:40] = True
    board[0] = True
    new, lines = eng.clear(board)
    assert lines == 1
    assert not new[30:40].any()
    assert new[0]


def test_clear_row_and_column_at_the_same_time():
    eng = Engine(10, ["dot"])
    board = empty_board()
    board[50:60] = True
    board[7::10] = True
    new, lines = eng.clear(board)
    assert lines == 2
    assert not new.any()


def test_clear_does_not_modify_the_input():
    eng = Engine(10, ["dot"])
    board = empty_board()
    board[0:10] = True
    eng.clear(board)
    assert board[0:10].all()


def test_score():
    assert Engine.score(3, 0) == 3
    assert Engine.score(5, 1) == 15
    assert Engine.score(1, 2) == 31
    assert Engine.score(9, 3) == 69
