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
        engine = env.engine
        actions, lines = [], []

        for slot, k in enumerate(env.hand):
            if k is None:
                continue

            ok = engine.valid(env.board, k)

            actions.append(slot * env.n + engine.anchors[k][ok])
            lines.append(engine.clear(engine.masks[k][ok] | env.board)[1])

        actions, lines = np.concatenate(actions), np.concatenate(lines)
        return int(self.rng.choice(actions[lines == lines.max()]))


class HeuristicAgent:

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def act(self, env, info):
        engine = env.engine
        actions, values = [], []

        for slot, k in enumerate(env.hand):
            if k is None:
                continue

            ok = engine.valid(env.board, k)
            new_boards, lines = engine.clear(engine.masks[k][ok] | env.board)

            n_places = engine.count_places(new_boards)
            mobility = np.log1p(n_places).mean(axis=1)

            blocked = (n_places == 0).mean(axis=1)
            rest = [j for s, j in enumerate(env.hand) if j is not None and s != slot]
            stuck = (n_places[:, rest] == 0).sum(axis=1)

            value = lines + mobility - 3 * blocked - 10 * stuck

            actions.append(slot * env.n + engine.anchors[k][ok])
            values.append(value)

        actions, values = np.concatenate(actions), np.concatenate(values)
        return int(self.rng.choice(actions[values == values.max()]))
