from tenten.agents.tabular import TabularQAgent
from tenten.game.env import TenTenEnv


def make_env():
    return TenTenEnv(size=5, piece_names=["small_L"], hand_size=1, max_steps=200)


def test_choose_only_valid_actions():
    env = make_env()
    agent = TabularQAgent(env.action_space.n)
    obs, info = env.reset(seed=0)
    for _ in range(50):
        action = agent.choose(agent.key(env), info["action_mask"], epsilon=0.5)
        assert info["action_mask"][action]
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()


def test_learn_updates_q():
    env = make_env()
    agent = TabularQAgent(env.action_space.n, alpha=0.5)
    obs, info = env.reset(seed=0)
    key = agent.key(env)
    agent.learn(
        key,
        action=0,
        reward=1.0,
        next_key=key,
        next_mask=info["action_mask"],
        terminated=True,
    )
    assert agent.Q[key][0] == 0.5


def test_same_board_same_key():
    env = make_env()
    env.reset(seed=0)
    key_1 = TabularQAgent.key(env)
    env.reset(seed=1)
    key_2 = TabularQAgent.key(env)
    assert key_1 == key_2
