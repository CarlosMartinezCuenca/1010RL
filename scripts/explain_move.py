"""Plays a few moves with the trained n-tuple agent and explains its next decision.

Usage:
    uv run python explain_move.py --model models/ntuple_last.npy --moves 20 --out docs/explain.png
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from tenten.agents.explain import cell_contributions, compare_moves
from tenten.agents.ntuple import NTupleAgent
from tenten.game.env import TenTenEnv


def describe(move, env):
    """'big_L_b at (row 3, col 5)' for an action."""
    slot, anchor = divmod(move["action"], env.n)
    piece = env.engine.piece_names[env.hand[slot]]
    row, col = divmod(anchor, env.size)
    return f"{piece} at (row {row}, col {col})"


def cells_of(move, env):
    """Cells covered by the piece of a move."""
    slot, anchor = divmod(move["action"], env.n)
    return env.engine.mask_at(env.hand[slot], anchor)


def heatmap(ax, values, board, title):
    """Diverging map centred on 0 (blue = adds value, red = removes value); dots = filled cells."""
    limit = np.abs(values).max() or 1.0
    ax.imshow(values.reshape(10, 10), cmap="RdBu", vmin=-limit, vmax=limit)
    rows, cols = np.divmod(np.flatnonzero(board), 10)
    ax.scatter(cols, rows, s=12, c="black")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/10x10_all/ntuple_seed0_best.npy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--moves", type=int, default=20, help="moves to play before explaining"
    )
    parser.add_argument("--out", default="docs/explain.png")
    args = parser.parse_args()

    # 1. Play some moves to reach an interesting position
    env = TenTenEnv()
    agent = NTupleAgent()
    agent.load(args.model)
    obs, info = env.reset(seed=args.seed)
    for _ in range(args.moves):
        obs, reward, terminated, truncated, info = env.step(agent.act(env, info))
        if terminated or truncated:
            raise SystemExit(
                "The game ended before reaching that move: try fewer --moves"
            )

    # 2. Explain the next decision
    exp = compare_moves(agent, env)
    if exp is None:
        raise SystemExit("Every legal move gives the same board: nothing to compare")
    best, second = exp["best"], exp["second"]

    print(
        f"Chosen:    {describe(best, env):32} reward {best['reward']:5.2f} | "
        f"V {best['value']:6.2f} | score {best['score']:6.2f}"
    )
    print(
        f"Runner-up: {describe(second, env):32} reward {second['reward']:5.2f} | "
        f"V {second['value']:6.2f} | score {second['score']:6.2f}"
        + ("  (certain loss)" if second["certain_loss"] else "")
    )
    print(
        f"Gap: {best['score'] - second['score']:.2f} = reward gap "
        f"{best['reward'] - second['reward']:.2f} + value gap {exp['diff_map'].sum():.2f}"
    )

    # 3. Figure: the decision, why the chosen board is good, and what changes vs the runner-up
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))

    picture = env.board.astype(int)  # 0 empty, 1 filled
    picture[cells_of(second, env)] = 3  # runner-up
    picture[cells_of(best, env)] = 2  # chosen (drawn on top)
    colors = ListedColormap(["#e2e2dc", "#2a78d6", "#eb6834", "#a78bfa"])
    axes[0].imshow(picture.reshape(10, 10), cmap=colors, vmin=0, vmax=3)
    axes[0].set_title(
        f"Move {info['steps'] + 1}: where to place the piece?\n"
        "orange = chosen · purple = runner-up · blue = filled",
        fontsize=10,
    )
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    heatmap(
        axes[1],
        cell_contributions(agent.net, best["afterstate"]),
        best["afterstate"],
        "Why the chosen board is good\n"
        f"value each cell adds to the board (total {best['value']:.2f})",
    )
    heatmap(
        axes[2],
        exp["diff_map"],
        best["afterstate"],
        "Why it beats the runner-up\n"
        f"board value per cell, chosen − runner-up (total {exp['diff_map'].sum():+.2f})",
    )

    fig.suptitle(
        f"Chosen beats runner-up by {best['score'] - second['score']:+.2f}: "
        f"{best['reward'] - second['reward']:+.2f} in points now, "
        f"{exp['diff_map'].sum():+.2f} in value of the board it leaves\n"
        "Heatmaps (board after the chosen move): blue = cell adds value, "
        "red = cell removes value, dots = filled cells",
        fontsize=10,
        y=0.02,
        va="bottom",
    )
    fig.tight_layout(rect=(0, 0.11, 1, 1))  # leave room at the bottom for the suptitle
    fig.savefig(args.out, dpi=150)
    print(f"Figure saved to {args.out}")


if __name__ == "__main__":
    main()
