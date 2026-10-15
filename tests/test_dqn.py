import numpy as np

from agents.dqn import DQNAgent
from tenten.env import TenTenEnv


def make_agent(env):
    return DQNAgent(
        env.observation_space.shape[0],
        env.action_space.n,
        hidden=32,
        buffer_size=1_000,
        batch_size=16,
        seed=0,
    )


def test_choose_only_valid_actions():
    env = TenTenEnv()
    agent = make_agent(env)
    obs, info = env.reset(seed=0)
    for _ in range(50):
        action = agent.choose(obs, info["action_mask"], epsilon=0.0)
        assert info["action_mask"][action]
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()


def test_learn_runs_and_returns_a_number():
    env = TenTenEnv()
    agent = make_agent(env)
    obs, info = env.reset(seed=0)
    for _ in range(100):
        action = agent.choose(obs, info["action_mask"], epsilon=1.0)
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.remember(obs, action, reward, next_obs, info["action_mask"], terminated)
        obs = next_obs
        if terminated or truncated:
            obs, info = env.reset()
    loss = agent.learn()
    assert np.isfinite(loss)


def test_save_and_load(tmp_path):
    env = TenTenEnv()
    agent = make_agent(env)
    path = tmp_path / "dqn.pt"
    agent.save(path)

    other = DQNAgent(
        env.observation_space.shape[0],
        env.action_space.n,
        hidden=32,
        buffer_size=1_000,
        batch_size=16,
        seed=123,
    )
    other.load(path)
    obs, info = env.reset(seed=0)
    assert agent.choose(obs, info["action_mask"]) == other.choose(
        obs, info["action_mask"]
    )
