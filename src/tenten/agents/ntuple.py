"""N-tuple network + TD(0) on afterstates (the board right after a move and its line clears).

V(board) is the sum of one weight per tuple (a fixed group of cells), indexed by the
pattern of those cells. The 8 rotations/reflections of a tuple share the same weights.
"""

import numpy as np

from tenten.game.engine import Engine

SHAPES = [
    (1, None),
    (None, 1),
    (2, 2),
    (3, 3),
    (2, 3),
    (3, 2),
]  # (height, width); None = whole board


def symmetries(size):
    """The 8 symmetries of the square as (row, col) -> (row, col) maps."""
    m = size - 1
    return [
        lambda r, c: (r, c),
        lambda r, c: (c, m - r),
        lambda r, c: (m - r, m - c),
        lambda r, c: (m - c, r),
        lambda r, c: (c, r),
        lambda r, c: (m - r, c),
        lambda r, c: (m - c, m - r),
        lambda r, c: (r, m - c),
    ]


class NTupleNetwork:
    """Value function V(board) as a sum of lookup tables."""

    def __init__(self, size=10):
        self.size = size
        tuples = []
        table_sizes = []
        seen = set()

        for h, w in SHAPES:
            h = size if h is None else h
            w = size if w is None else w
            for row in range(size - h + 1):
                for col in range(size - w + 1):
                    cells = [
                        (row + i) * size + (col + j) for i in range(h) for j in range(w)
                    ]
                    if frozenset(cells) in seen:
                        continue  # it is a rotation/reflection of a tuple that already has a table

                    table = len(table_sizes)
                    table_sizes.append(len(cells))

                    images = set()
                    for g in symmetries(size):
                        image = tuple(
                            g(*divmod(c, size))[0] * size + g(*divmod(c, size))[1]
                            for c in cells
                        )
                        images.add(image)
                        seen.add(frozenset(image))
                    for image in images:
                        tuples.append((image, table))

        # A table of k cells has 2^k weights; all tables live in one vector, one after another
        offsets = np.cumsum([0] + [2**k for k in table_sizes])
        self.weights = np.zeros(offsets[-1])

        # powers[cell, t] = 2^j if cell is the j-th cell of tuple t, so the pattern index is a matmul
        self.powers = np.zeros((size * size, len(tuples)), dtype=np.float32)
        for t, (cells, table) in enumerate(tuples):
            self.powers[list(cells), t] = 2.0 ** np.arange(len(cells))
        self.starts = offsets[[table for _, table in tuples]]
        self.n_tuples = len(tuples)

    def indices(self, boards):
        """(B, n) boards -> (B, n_tuples) indices of the active weight of each tuple."""
        patterns = boards.astype(np.float32) @ self.powers  # exact: at most 2^10 - 1
        return patterns.astype(np.int64) + self.starts

    def value(self, boards):
        """V of a batch of boards."""
        return self.weights[self.indices(boards)].sum(axis=1)

    def update(self, board, target, alpha):
        """Moves V(board) a fraction alpha towards target; returns the TD error."""
        idx = self.indices(board[None, :])[0]
        error = target - self.weights[idx].sum()
        np.add.at(
            self.weights, idx, alpha * error / len(idx)
        )  # add.at: an index can repeat
        return error


def place_everywhere(env, board, piece):
    """Every legal placement of piece: (anchors, resulting boards after clearing, lines)."""
    engine = env.engine
    ok = engine.valid(board, piece)
    boards, lines = engine.clear(board | engine.masks[piece][ok])
    return engine.anchors[piece][ok], boards, lines


def fits_somewhere(env, boards, piece):
    """For each board, whether piece fits anywhere."""
    masks = env.engine.masks[piece].astype(np.float32)
    overlaps = boards.astype(np.float32) @ masks.T
    return (overlaps == 0).any(axis=1)


class NTupleAgent:
    """Picks the move maximising r + gamma * V(afterstate)."""

    def __init__(
        self,
        size=10,
        alpha=0.1,
        gamma=0.95,
        reward_scale=0.1,
        loss_penalty=0.0,
        seed=0,
    ):
        self.net = NTupleNetwork(size)
        self.alpha = alpha
        self.gamma = gamma

        self.reward_scale = reward_scale
        self.loss_penalty = loss_penalty
        self.rng = np.random.default_rng(seed)

    def candidates(self, env, board=None, hand=None):
        """All legal moves with their slot, afterstate, reward and r + gamma * V."""
        board = env.board if board is None else board
        hand = env.hand if hand is None else hand

        actions, slots, afterstates, rewards = [], [], [], []
        for slot, piece in enumerate(hand):
            if piece is None:
                continue
            anchors, boards, lines = place_everywhere(env, board, piece)
            points = Engine.score(int(env.engine.pieces[piece].sum()), lines)

            actions.append(slot * env.n + anchors)
            slots.append(np.full(len(anchors), slot))
            afterstates.append(boards)

            rewards.append(self.reward_scale * points)

        actions = np.concatenate(actions)
        slots = np.concatenate(slots)
        afterstates = np.concatenate(afterstates)
        rewards = np.concatenate(rewards)

        values = self.net.value(afterstates)

        # If none of the remaining hand pieces fits, the game is certainly lost
        for slot in np.unique(slots):
            rest = [k for s, k in enumerate(hand) if k is not None and s != slot]
            if not rest:
                continue  # last piece: a new random hand comes, nothing is certain
            this_slot = slots == slot
            any_fits = np.zeros(this_slot.sum(), dtype=bool)
            for k in rest:
                any_fits |= fits_somewhere(env, afterstates[this_slot], k)
            values[np.flatnonzero(this_slot)[~any_fits]] = self.loss_penalty

        scores = rewards + self.gamma * values
        return actions, slots, afterstates, rewards, scores

    def choose(self, env, epsilon=0.0):
        """For training: epsilon-greedy (action, afterstate, reward)."""
        actions, _, afterstates, rewards, scores = self.candidates(env)
        if self.rng.random() < epsilon:
            i = self.rng.integers(len(actions))
        else:
            i = np.argmax(scores)
        return int(actions[i]), afterstates[i], float(rewards[i])

    def act(self, env, info):
        """Greedy move, same interface as the baselines."""
        actions, _, _, _, scores = self.candidates(env)
        return int(actions[np.argmax(scores)])

    def learn(self, afterstate, target):
        """TD update of one afterstate; returns the error."""
        return self.net.update(afterstate, target, self.alpha)

    def value(self, afterstate):
        """V of a single afterstate."""
        return float(self.net.value(afterstate[None, :])[0])

    def save(self, path):
        """Saves the weights (.npy)."""
        np.save(path, self.net.weights)

    def load(self, path):
        """Loads the weights (.npy)."""
        self.net.weights = np.load(path)


class SearchAgent:
    """One-step lookahead with a trained NTupleAgent, using the known pieces left in hand."""

    def __init__(self, agent, top_k=12):
        self.agent = agent
        self.top_k = top_k

    def act(self, env, info):
        """Re-scores the top_k moves with the best follow-up move of the remaining pieces."""
        a = self.agent
        actions, slots, afterstates, rewards, scores = a.candidates(env)
        best_action, best_score = actions[np.argmax(scores)], -np.inf
        for i in np.argsort(-scores)[: self.top_k]:
            rest = [k if s != slots[i] else None for s, k in enumerate(env.hand)]
            if all(k is None for k in rest):
                score = scores[i]  # last piece: the next hand is random, no lookahead
            else:
                *_, next_scores = a.candidates(env, board=afterstates[i], hand=rest)
                best_next = next_scores.max() if len(next_scores) else a.loss_penalty
                score = rewards[i] + a.gamma * best_next
            if score > best_score:
                best_action, best_score = actions[i], score
        return int(best_action)
