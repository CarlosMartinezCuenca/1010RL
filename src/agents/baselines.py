import numpy as np


class RandomAgent:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def act(self, env, info):
        valid_actions = np.flatnonzero(info["action_mask"])
        return int(self.rng.choice(valid_actions))


class GreedyAgent:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def act(self, env, info):
        best_actions, best_lines = [], -1

        for action in np.flatnonzero(info["action_mask"]):
            slot, anchor = divmod(int(action), env.n)
            piece = env.hand[slot]

            cells = env.engine.mask_at(piece, anchor)
            _, lines = env.engine.clear(env.board | cells)

            if lines > best_lines:
                best_actions, best_lines = [int(action)], lines
            elif lines == best_lines:
                best_actions.append(int(action))

        return int(self.rng.choice(best_actions))
