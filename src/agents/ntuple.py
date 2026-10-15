"""Agente final: red de n-tuplas + TD(0) sobre afterstates (+ búsqueda opcional).

- Afterstate = el tablero justo DESPUÉS de colocar una pieza (y borrar líneas).
  Colocar es determinista, así que el agente simula cada jugada válida y elige
  la que maximiza   r + γ · V(afterstate).
- V(tablero) la da una red de n-tuplas: muchos grupos pequeños de casillas (filas,
  columnas, cuadrados...) y una tabla de pesos por grupo. V = suma de los pesos
  que "se encienden" con el patrón de cada grupo.
- Simetría: el tablero girado o reflejado vale lo mismo, así que las 8 versiones
  de cada grupo comparten la MISMA tabla de pesos.
"""

import numpy as np

from tenten.engine import Engine

# Formas de los grupos de casillas (alto, ancho). None = todo el tablero.
SHAPES = [(1, None), (None, 1), (2, 2), (3, 3), (2, 3), (3, 2)]


def symmetries(size):
    """Las 8 simetrías del cuadrado: (fila, col) -> (fila', col')."""
    m = size - 1
    return [
        lambda r, c: (r, c),  # identidad
        lambda r, c: (c, m - r),  # giro 90°
        lambda r, c: (m - r, m - c),  # giro 180°
        lambda r, c: (m - c, r),  # giro 270°
        lambda r, c: (c, r),  # reflejo diagonal
        lambda r, c: (m - r, c),  # reflejo vertical
        lambda r, c: (m - c, m - r),  # reflejo antidiagonal
        lambda r, c: (r, m - c),  # reflejo horizontal
    ]


class NTupleNetwork:
    def __init__(self, size=10):
        self.size = size
        tuples = []  # lista de (casillas, número de tabla)
        table_sizes = []  # nº de casillas de cada tabla
        seen = set()  # grupos ya cubiertos (como imagen de otro)

        for h, w in SHAPES:
            h = size if h is None else h
            w = size if w is None else w
            for row in range(size - h + 1):
                for col in range(size - w + 1):
                    cells = [
                        (row + i) * size + (col + j) for i in range(h) for j in range(w)
                    ]
                    if frozenset(cells) in seen:
                        continue  # es un giro/reflejo de otro grupo: ya tiene tabla
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

        # Cada tabla de k casillas tiene 2^k pesos (un peso por cada patrón posible).
        # Todas las tablas van seguidas en un único vector de pesos; offset = dónde empieza cada una.
        offsets = np.cumsum([0] + [2**k for k in table_sizes])
        self.weights = np.zeros(offsets[-1])

        # powers[casilla, tupla] = 2^j si la casilla es la j-ésima de la tupla (si no, 0).
        # Así el índice de todas las tuplas sale de una sola multiplicación de matrices.
        self.powers = np.zeros((size * size, len(tuples)), dtype=np.float32)
        for t, (cells, table) in enumerate(tuples):
            self.powers[list(cells), t] = 2.0 ** np.arange(len(cells))
        self.starts = offsets[[table for _, table in tuples]]
        self.n_tuples = len(tuples)

    def indices(self, boards):
        """boards (B, n) -> (B, n_tuples): qué peso se enciende en cada tupla.
        El patrón de casillas se lee como un número binario."""
        patterns = boards.astype(np.float32) @ self.powers  # exacto: como mucho 2^10 - 1
        return patterns.astype(np.int64) + self.starts

    def value(self, boards):
        """V de varios tableros a la vez."""
        return self.weights[self.indices(boards)].sum(axis=1)

    def update(self, board, target, alpha):
        """Mueve V(board) una fracción alpha hacia target (TD). Devuelve el error."""
        idx = self.indices(board[None, :])[0]
        error = target - self.weights[idx].sum()
        np.add.at(
            self.weights, idx, alpha * error / len(idx)
        )  # add.at: un índice puede repetirse
        return error


def place_everywhere(env, board, piece):
    """Todas las colocaciones válidas de `piece` en `board`, a la vez.
    Devuelve (anclas, tableros resultantes ya limpiados, líneas de cada uno)."""
    engine = env.engine
    ok = engine.valid(board, piece)
    boards, lines = engine.clear(board | engine.masks[piece][ok])
    return engine.anchors[piece][ok], boards, lines


def fits_somewhere(env, boards, piece):
    """Para cada tablero: ¿cabe `piece` en algún sitio?"""
    masks = env.engine.masks[piece].astype(np.float32)
    overlaps = (
        boards.astype(np.float32) @ masks.T
    )  # (B, posiciones): casillas que chocan
    return (overlaps == 0).any(axis=1)


class NTupleAgent:
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
        # Recompensa = puntos de la jugada (la misma que da el entorno), escalada
        self.reward_scale = reward_scale
        # Valor de perder. 0 = sin castigo extra (perder ya cuesta los puntos futuros);
        # un valor negativo es un experimento de reward shaping aparte
        self.loss_penalty = loss_penalty
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ jugar
    def candidates(self, env, board=None, hand=None):
        """Todas las jugadas válidas, con su afterstate y su puntuación r + γ·V."""
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
        # Si ninguna de las piezas que quedan en la mano cabe: game over seguro
        for slot in np.unique(slots):
            rest = [k for s, k in enumerate(hand) if k is not None and s != slot]
            if not rest:
                continue  # era la última pieza: llega mano nueva, no se sabe
            this_slot = slots == slot
            any_fits = np.zeros(this_slot.sum(), dtype=bool)
            for k in rest:
                any_fits |= fits_somewhere(env, afterstates[this_slot], k)
            values[np.flatnonzero(this_slot)[~any_fits]] = self.loss_penalty

        scores = rewards + self.gamma * values
        return actions, slots, afterstates, rewards, scores

    def choose(self, env, epsilon=0.0):
        """Para entrenar: devuelve (acción, afterstate, recompensa)."""
        actions, _, afterstates, rewards, scores = self.candidates(env)
        if self.rng.random() < epsilon:
            i = self.rng.integers(len(actions))
        else:
            i = np.argmax(scores)
        return int(actions[i]), afterstates[i], float(rewards[i])

    def act(self, env, info):
        """Misma forma que las baselines, para poder usar play_games."""
        actions, _, _, _, scores = self.candidates(env)
        return int(actions[np.argmax(scores)])

    # ------------------------------------------------------------------ aprender
    def learn(self, afterstate, target):
        return self.net.update(afterstate, target, self.alpha)

    def value(self, afterstate):
        return float(self.net.value(afterstate[None, :])[0])

    # ------------------------------------------------------------------ guardar
    def save(self, path):
        np.save(path, self.net.weights)

    def load(self, path):
        self.net.weights = np.load(path)


class SearchAgent:
    """Búsqueda de un paso con la red ya entrenada (no aprende nada nuevo).

    Las piezas que quedan en la mano son CONOCIDAS, así que para las mejores
    jugadas se mira también cuál sería la mejor jugada siguiente con esas piezas.
    """

    def __init__(self, agent, top_k=12):
        self.agent = agent
        self.top_k = top_k  # solo se mira más a fondo las 12 mejores jugadas

    def act(self, env, info):
        a = self.agent
        actions, slots, afterstates, rewards, scores = a.candidates(env)
        best_action, best_score = actions[np.argmax(scores)], -np.inf
        for i in np.argsort(-scores)[: self.top_k]:
            rest = [k if s != slots[i] else None for s, k in enumerate(env.hand)]
            if all(k is None for k in rest):
                score = scores[
                    i
                ]  # era la última pieza: llega mano nueva (azar), no se mira más
            else:
                *_, next_scores = a.candidates(env, board=afterstates[i], hand=rest)
                best_next = next_scores.max() if len(next_scores) else a.loss_penalty
                score = rewards[i] + a.gamma * best_next
            if score > best_score:
                best_action, best_score = actions[i], score
        return int(best_action)
