"""Compares the width of the search (top_k) on the same games, with a trained model.

uv run python scripts/compare_top_k.py
uv run python scripts/compare_top_k.py --top-k 5 10 20 --games 50
"""

import argparse
import json
import os
import time

# One thread per process (set before importing NumPy), as in experiment.py
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402

from tenten.agents.registry import load_agent  # noqa: E402
from tenten.evaluation import play_games  # noqa: E402
from tenten.game.env import TenTenEnv  # noqa: E402


def main():
    """Plays the same games with each top_k and compares them with the first one."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/10x10_all/ntuple_seed0_best.npy")
    parser.add_argument("--top-k", type=int, nargs="+", default=[10, 5, 20])
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=5000)
    # Same games as the final evaluation of experiment.py
    parser.add_argument("--seed", type=int, default=20_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default="results/10x10_all/search_top_k.json")
    args = parser.parse_args()

    env = TenTenEnv(max_steps=args.max_steps)
    games = {}
    for k in args.top_k:
        agent = load_agent("ntuple_search", env, args.model, top_k=k)
        start = time.perf_counter()
        games[k] = play_games(agent, env, args.games, args.seed, args.workers)
        minutes = (time.perf_counter() - start) / 60
        score, steps = np.array(games[k]["score"]), np.array(games[k]["steps"])
        capped = 100 * np.mean(steps >= args.max_steps)
        print(
            f"top_k {k:>3}: mean {score.mean():8.1f} | "
            f"median {np.median(score):8.1f} | "
            f"{capped:3.0f}% reach the cap | {minutes:5.1f} min",
            flush=True,
        )

    # Game-by-game difference with the first top_k, with a bootstrap 95% interval
    first = args.top_k[0]
    rng = np.random.default_rng(0)
    for k in args.top_k[1:]:
        diff = np.array(games[k]["score"]) - np.array(games[first]["score"])
        means = rng.choice(diff, size=(10_000, len(diff))).mean(axis=1)
        low, high = np.percentile(means, [2.5, 97.5])
        print(
            f"top_k {k} vs {first}: {diff.mean():+8.1f} points, "
            f"95% CI [{low:+.1f}, {high:+.1f}], "
            f"better in {(diff > 0).sum()}/{len(diff)}"
        )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({str(k): g for k, g in games.items()}, f, indent=2)
    print(f"Per-game results saved to {args.out}")


if __name__ == "__main__":
    main()
