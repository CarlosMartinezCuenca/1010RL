"""Tabular Q-learning: one Q-value per (state, action) actually visited."""

import pickle

import numpy as np


class TabularQAgent:
    """Q-table stored as a dict: state key -> array of Q-values."""

    def __init__(self, n_actions, alpha=0.1, gamma=0.99, seed=0):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)
        self.Q = {}

        # Q of unseen states; read-only so nobody modifies the shared array
        self.zeros = np.zeros(n_actions)
        self.zeros.setflags(write=False)

    @staticmethod
    def key(env):
        """State = board + hand (255 marks an empty slot)."""
        hand = bytes(255 if k is None else k for k in env.hand)
        return env.board.tobytes() + hand

    def q_values(self, key):
        """Q-values of a state; reading does not add it to the table."""
        return self.Q.get(key, self.zeros)

    def choose(self, key, mask, epsilon=0.0):
        """Epsilon-greedy over legal actions, ties broken at random."""
        valid = np.flatnonzero(mask)
        if self.rng.random() < epsilon:
            return int(self.rng.choice(valid))
        q = self.q_values(key)[valid]
        best = valid[q == q.max()]
        return int(self.rng.choice(best))

    def act(self, env, info):
        """Greedy move, same interface as the baselines."""
        return self.choose(self.key(env), info["action_mask"], epsilon=0.0)

    def learn(self, key, action, reward, next_key, next_mask, terminated):
        """Q-learning update: Q(s, a) += alpha * (r + gamma * max Q(s', a') - Q(s, a))."""
        if terminated:
            target = reward
        else:
            target = reward + self.gamma * self.q_values(next_key)[next_mask].max()
        if key not in self.Q:
            self.Q[key] = np.zeros(self.n_actions)
        q = self.Q[key]
        q[action] += self.alpha * (target - q[action])

    def save(self, path):
        """Saves the Q-table (pickle)."""
        with open(path, "wb") as f:
            pickle.dump(self.Q, f)

    def load(self, path):
        """Loads the Q-table (pickle)."""
        with open(path, "rb") as f:
            self.Q = pickle.load(f)
