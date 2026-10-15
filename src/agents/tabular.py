import pickle

import numpy as np


class TabularQAgent:
    def __init__(self, n_actions, alpha=0.1, gamma=0.99, seed=0):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)
        self.Q = {}

        self.zeros = np.zeros(n_actions)
        self.zeros.setflags(write=False)

    @staticmethod
    def key(env):
        hand = bytes(255 if k is None else k for k in env.hand)
        return env.board.tobytes() + hand

    def q_values(self, key):
        return self.Q.get(key, self.zeros)

    def choose(self, key, mask, epsilon=0.0):
        valid = np.flatnonzero(mask)
        if self.rng.random() < epsilon:
            return int(self.rng.choice(valid))
        q = self.q_values(key)[valid]
        best = valid[q == q.max()]
        return int(self.rng.choice(best))

    def act(self, env, info):
        return self.choose(self.key(env), info["action_mask"], epsilon=0.0)

    def learn(self, key, action, reward, next_key, next_mask, terminated):
        if terminated:
            target = reward
        else:
            target = reward + self.gamma * self.q_values(next_key)[next_mask].max()
        if key not in self.Q:
            self.Q[key] = np.zeros(self.n_actions)
        q = self.Q[key]
        q[action] += self.alpha * (target - q[action])

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.Q, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.Q = pickle.load(f)
