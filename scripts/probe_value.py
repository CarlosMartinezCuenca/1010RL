"""Probing the learned value function: what does V(board) actually measure?

Plays some games with the trained agent, records every board it leaves (the afterstates)
and checks how well interpretable features explain V.

Usage:
    uv run python scripts/probe_value.py
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from tenten.agents.ntuple import NTupleAgent
from tenten.agents.probing import FEATURES, board_features, fit_probe
from tenten.game.env import TenTenEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/10x10_all/ntuple_seed0_best.npy")
    parser.add_argument("--games", type=int, default=20)
    # Different games from training and evaluation
    parser.add_argument("--seed", type=int, default=20_000)
    parser.add_argument("--out", default="docs/probing.png")
    args = parser.parse_args()

    # 1. Collect the boards the agent leaves after each move, with their V and features
    env = TenTenEnv()
    agent = NTupleAgent()
    agent.load(args.model)
    features, values = [], []
    for game in range(args.games):
        obs, info = env.reset(seed=args.seed + game)
        done = False
        while not done:
            obs, reward, terminated, truncated, info = env.step(agent.act(env, info))
            done = terminated or truncated
            # env.board is now the afterstate of the move just played
            if not terminated:
                features.append(board_features(env.engine, env.board))
                values.append(agent.value(env.board))
    features, values = np.array(features), np.array(values)

    # 2. Fit the probe
    probe = fit_probe(features, values)
    print(f"{len(values)} boards from {args.games} games")
    print(f"{'feature':>18} | correlation with V | standardised coefficient")
    for name, r, c in zip(FEATURES, probe["correlations"], probe["coefficients"]):
        print(f"{name:>18} | {r:+18.2f} | {c:+24.2f}")
    print(f"R² of the linear fit: {probe['r2']:.2f}")

    # 3. Figure: correlation of V with each feature
    fig, ax = plt.subplots(figsize=(7, 3.2))
    r = probe["correlations"]
    # Blue = V goes up with the feature, red = V goes down
    colors = np.where(r >= 0, "#2a78d6", "#d64545")
    y = np.arange(len(FEATURES))[::-1]
    ax.barh(y, r, color=colors, height=0.5)
    for yi, ri in zip(y, r):
        ax.text(
            ri + (0.03 if ri >= 0 else -0.03),
            yi,
            f"{ri:+.2f}",
            va="center",
            ha="left" if ri >= 0 else "right",
            fontsize=9,
        )
    ax.set_yticks(y, [name.replace("_", " ") for name in FEATURES])
    ax.axvline(0, color="#6b7280", lw=0.8)
    ax.set_xlim(-1.15, 1.15)
    ax.set_xlabel("correlation with the learned value V(board)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(
        f"What V measures · linear fit on these 5 features: R² = {probe['r2']:.2f}",
        fontsize=10,
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Figure saved to {args.out}")

    # Mobility vs V among boards with a similar number of filled cells
    empty = features[:, 0]
    for low in np.arange(0.3, 0.8, 0.1):
        same = (empty >= low) & (empty < low + 0.1)
        if same.sum() > 50:
            r = np.corrcoef(features[same, 1], values[same])[0, 1]
            print(
                f"empty {low:.1f}-{low + 0.1:.1f}: {same.sum():5d} boards | corr(mobility, V) = {r:+.2f}"
            )


if __name__ == "__main__":
    main()
