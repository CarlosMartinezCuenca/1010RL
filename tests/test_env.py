import numpy as np

from tenten.env import TenTenEnv
from tenten.pieces import PIECES

NAMES = list(PIECES)


def test_reset():
    env = TenTenEnv()
    obs, info = env.reset(seed=0)
    assert obs.shape == (400,)
    assert obs[:100].sum() == 0
    assert all(k is not None for k in env.hand)
    assert info["action_mask"].shape == (300,)
    assert info["action_mask"].any()


def test_same_seed_same_hand():
    a, b = TenTenEnv(), TenTenEnv()
    a.reset(seed=42)
    b.reset(seed=42)
    assert a.hand == b.hand


def test_step_places_the_piece():
    env = TenTenEnv()
    env.reset(seed=0)
    env.hand = [NAMES.index("square2"), None, None]
    env.step(0 * 100 + 0)
    assert set(np.flatnonzero(env.board)) == {0, 1, 10, 11}


def test_step_clears_a_line_and_gives_reward():
    env = TenTenEnv()
    env.reset(seed=0)
    env.board[0:9] = True
    env.hand = [NAMES.index("dot"), NAMES.index("dot"), NAMES.index("dot")]
    obs, reward, terminated, truncated, info = env.step(9)
    assert reward == 1.0
    assert not env.board.any()
    assert info["lines"] == 1
    assert info["score"] == 1 + 10


def test_used_slot_is_empty_and_masked():
    env = TenTenEnv()
    _, info = env.reset(seed=1)
    a = int(np.flatnonzero(info["action_mask"][:100])[0])  # una jugada del hueco 0
    _, _, _, _, info = env.step(a)
    assert env.hand[0] is None
    assert not info["action_mask"][:100].any()  # ese hueco ya no se puede usar


def test_new_hand_after_three_moves():
    env = TenTenEnv()
    _, info = env.reset(seed=2)
    for _ in range(3):
        a = int(np.flatnonzero(info["action_mask"])[0])
        _, _, _, _, info = env.step(a)
    assert all(k is not None for k in env.hand)


def test_invalid_action_ends_the_game():
    env = TenTenEnv()
    env.reset(seed=0)
    env.board[0] = True
    env.hand = [NAMES.index("dot"), None, None]
    _, reward, terminated, _, _ = env.step(0)  # casilla ocupada
    assert terminated and reward == 0.0
    env.reset(seed=0)
    env.hand = [NAMES.index("dot"), None, None]
    _, _, terminated, _, _ = env.step(100 + 5)  # hueco 1 vacío
    assert terminated


def test_game_over_when_nothing_fits():
    env = TenTenEnv()
    env.reset(seed=0)
    # Tablero lleno salvo la diagonal y la casilla (0,1): ninguna fila ni
    # columna está completa, y no hay ningún hueco de 3x3.
    env.board[:] = True
    env.board[[i * 10 + i for i in range(10)]] = False
    env.board[1] = False
    sq3 = NAMES.index("square3")
    env.hand = [NAMES.index("dot"), sq3, sq3]
    _, _, terminated, truncated, info = env.step(
        1
    )  # el punto en (0,1): no completa nada
    assert terminated and not truncated  # los 3x3 no caben
    assert not info["action_mask"].any()


def test_truncated_at_max_steps():
    env = TenTenEnv(max_steps=2)
    _, info = env.reset(seed=3)
    for i in range(2):
        a = int(np.flatnonzero(info["action_mask"])[0])
        _, _, terminated, truncated, info = env.step(a)
    assert truncated and not terminated


def test_random_games_run_to_the_end():
    env = TenTenEnv()
    rng = np.random.default_rng(0)
    for seed in range(20):
        _, info = env.reset(seed=seed)
        while True:
            a = rng.choice(np.flatnonzero(info["action_mask"]))
            obs, reward, terminated, truncated, info = env.step(a)
            assert env.observation_space.contains(obs)
            if terminated or truncated:
                break
