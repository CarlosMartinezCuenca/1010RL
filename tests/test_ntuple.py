import numpy as np

from agents.ntuple import NTupleAgent, NTupleNetwork, SearchAgent, place_everywhere
from tenten.env import TenTenEnv
from tenten.pieces import PIECES

DOT = list(PIECES).index("dot")


def test_network_size():
    net = NTupleNetwork(10)
    assert len(net.weights) == 11760
    assert net.value(np.zeros((1, 100), dtype=bool))[0] == 0.0


def test_symmetric_boards_have_same_value():
    net = NTupleNetwork(10)
    net.weights = np.random.default_rng(0).normal(size=len(net.weights))
    board = np.random.default_rng(1).random((10, 10)) < 0.4
    rotated = np.rot90(board)
    mirrored = np.fliplr(board)
    values = net.value(np.array([board.ravel(), rotated.ravel(), mirrored.ravel()]))
    assert np.allclose(values, values[0])


def test_update_moves_value_towards_target():
    net = NTupleNetwork(10)
    board = np.zeros(100, dtype=bool)
    board[:5] = True
    net.update(board, target=1.0, alpha=0.01)
    value = net.value(board[None, :])[0]
    assert 0.0 < value < 1.0


def test_place_everywhere_clears_lines():
    env = TenTenEnv()
    env.reset(seed=0)
    board = np.zeros(100, dtype=bool)
    board[0:9] = True
    anchors, boards, lines = place_everywhere(env, board, DOT)
    i = np.flatnonzero(anchors == 9)[0]
    assert lines[i] == 1 and not boards[i].any()


def test_agents_only_pick_valid_actions():
    env = TenTenEnv()
    agent = NTupleAgent()
    for player in [agent, SearchAgent(agent)]:
        obs, info = env.reset(seed=0)
        for _ in range(30):
            action = player.act(env, info)
            assert info["action_mask"][action]
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()


def test_simulated_move_matches_the_environment():
    # El agente simula cada jugada: su afterstate y su recompensa deben ser los reales
    env = TenTenEnv()
    agent = NTupleAgent()
    obs, info = env.reset(seed=0)
    for _ in range(50):
        action, afterstate, reward = agent.choose(env)
        obs, env_reward, terminated, truncated, info = env.step(action)
        assert np.array_equal(afterstate, env.board)
        assert np.isclose(reward, agent.reward_scale * env_reward)
        if terminated or truncated:
            obs, info = env.reset()
