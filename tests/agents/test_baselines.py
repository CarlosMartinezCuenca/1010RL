"""Tests for the baseline agents."""

import numpy as np

from tenten.agents.baselines import GreedyAgent, HeuristicAgent, RandomAgent
from tenten.evaluation import play_games
from tenten.game.env import TenTenEnv
from tenten.game.pieces import PIECES

DOT = list(PIECES).index("dot")


def test_random_only_picks_valid_actions():
    env = TenTenEnv()
    agent = RandomAgent(seed=0)
    obs, info = env.reset(seed=0)
    for _ in range(50):
        action = agent.act(env, info)
        assert info["action_mask"][action]
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()


def test_greedy_takes_the_line():
    env = TenTenEnv()
    env.reset(seed=0)
    env.board[:] = False
    env.board[0:9] = True
    env.hand = [DOT, None, None]
    info = {"action_mask": env.action_mask()}
    action = GreedyAgent(seed=0).act(env, info)
    assert action == 0 * env.n + 9  # slot 0, cell 9


def test_greedy_beats_random():
    env = TenTenEnv()
    random_scores = play_games(RandomAgent(seed=0), env, n_games=20)["score"]
    greedy_scores = play_games(GreedyAgent(seed=0), env, n_games=20)["score"]
    assert np.mean(greedy_scores) > np.mean(random_scores)


def test_heuristic_takes_the_line():
    env = TenTenEnv()
    env.reset(seed=0)
    env.board[:] = False
    env.board[0:9] = True
    env.hand = [DOT, None, None]
    info = {"action_mask": env.action_mask()}
    action = HeuristicAgent(seed=0).act(env, info)
    assert action == 0 * env.n + 9


def test_heuristic_beats_greedy():
    env = TenTenEnv()
    greedy_scores = play_games(GreedyAgent(seed=0), env, n_games=10)["score"]
    heuristic_scores = play_games(HeuristicAgent(seed=0), env, n_games=10)["score"]
    assert np.mean(heuristic_scores) > np.mean(greedy_scores)
