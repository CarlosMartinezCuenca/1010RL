import numpy as np

from tenten.agents.explain import cell_contributions, compare_moves
from tenten.agents.ntuple import NTupleAgent
from tenten.game.env import TenTenEnv


def random_agent():
    """N-tuple agent with random weights (no training needed to test the maths)."""
    agent = NTupleAgent()
    agent.net.weights = np.random.default_rng(0).normal(size=len(agent.net.weights))
    return agent


def test_contributions_add_up_to_value():
    agent = random_agent()
    board = np.random.default_rng(1).random(100) < 0.4
    contributions = cell_contributions(agent.net, board)
    assert contributions.shape == (100,)
    assert np.isclose(contributions.sum(), agent.value(board))


def test_score_gap_is_reward_gap_plus_value_gap():
    agent = random_agent()
    env = TenTenEnv()
    obs, info = env.reset(seed=0)
    exp = compare_moves(agent, env)
    best, second = exp["best"], exp["second"]
    assert best["score"] >= second["score"]
    assert not best["certain_loss"] and not second["certain_loss"]
    gap = best["score"] - second["score"]
    assert np.isclose(gap, best["reward"] - second["reward"] + exp["diff_map"].sum())
