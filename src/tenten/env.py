import numpy as np

import gymnasium as gym
from gymnasium import spaces

from tenten.engine import Engine
from tenten.pieces import PIECES


class TenTenEnv(gym.Env):
    def __init__(
        self,
        size: int = 10,
        piece_names: list[str] | None = None,
        hand_size: int = 3,
        max_steps: int = 1000,
    ):
        super().__init__()
        if piece_names is None:
            piece_names = list(PIECES)

        self.size = size
        self.n = size * size
        self.hand_size = hand_size
        self.max_steps = max_steps
        self.engine = Engine(size, piece_names)

        self.observation_space = spaces.Box(
            0, 1, shape=((1 + hand_size) * self.n,), dtype=np.uint8
        )
        self.action_space = spaces.Discrete(hand_size * self.n)

        self.board = np.zeros(self.n, dtype=bool)
        self.hand: list[int | None] = [None] * hand_size

        self.steps = 0
        self.lines = 0
        self.score = 0

    def _deal(self):
        n_pieces = len(self.engine.pieces)
        self.hand = [
            int(k) for k in self.np_random.integers(n_pieces, size=self.hand_size)
        ]

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(self.hand_size * self.n, dtype=bool)
        for slot, k in enumerate(self.hand):
            if k is None:
                continue
            ok = self.engine.valid(self.board, k)
            mask[slot * self.n + self.engine.anchors[k][ok]] = True
        return mask

    def _obs(self) -> np.ndarray:
        empty = np.zeros(self.n, dtype=np.uint8)  # hueco vacío -> plano de ceros
        planes = [self.board.astype(np.uint8)]
        planes += [empty if k is None else self.engine.planes[k] for k in self.hand]
        return np.concatenate(planes)

    def _info(self, mask: np.ndarray) -> dict:
        return {
            "action_mask": mask,
            "steps": self.steps,
            "lines": self.lines,
            "score": self.score,
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        self.board = np.zeros(self.n, dtype=bool)
        self.steps = 0
        self.lines = 0
        self.score = 0

        self._deal()
        return self._obs(), self._info(self.action_mask())

    def step(self, action: int):
        slot, anchor = divmod(int(action), self.n)

        k = self.hand[slot]
        m = None if k is None else self.engine.mask_at(k, anchor)
        if m is None or (m & self.board).any():
            return self._obs(), 0.0, True, False, self._info(self.action_mask())

        self.board, cleared = self.engine.clear(self.board | m)
        cleared = int(cleared)
        self.steps += 1
        self.lines += cleared
        points = Engine.score(int(m.sum()), cleared)
        self.score += points

        self.hand[slot] = None
        if all(h is None for h in self.hand):
            self._deal()

        mask = self.action_mask()
        terminated = not mask.any()
        truncated = (not terminated) and self.steps >= self.max_steps
        reward = float(points)  # recompensa = puntos de la jugada, la misma métrica que se evalúa

        return self._obs(), reward, terminated, truncated, self._info(mask)
