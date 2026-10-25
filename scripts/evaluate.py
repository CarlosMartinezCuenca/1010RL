"""Evaluates the baselines on the full 10x10 game (100 games each) and logs them to MLflow.

uv run python scripts/evaluate.py
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db      # http://127.0.0.1:5000
"""

import mlflow
import numpy as np

from tenten.agents.baselines import GreedyAgent, HeuristicAgent, RandomAgent
from tenten.evaluation import play_games
from tenten.game.env import TenTenEnv
from tenten.tracking import setup_mlflow


def main():
    """Plays and logs random, greedy and heuristic."""
    n_games = 100
    seed = 0
    env = TenTenEnv()

    setup_mlflow("baselines")

    for name, agent in [
        ("random", RandomAgent(seed=seed)),
        ("greedy", GreedyAgent(seed=seed)),
        ("heuristic", HeuristicAgent(seed=seed)),
    ]:
        results = play_games(agent, env, n_games)
        with mlflow.start_run(run_name=f"{name}_seed{seed}"):
            mlflow.log_params(
                {"agent": name, "n_games": n_games, "max_steps": env.max_steps}
            )
            mlflow.log_metrics(
                {
                    "mean_score": float(np.mean(results["score"])),
                    "std_score": float(np.std(results["score"])),
                    "mean_lines": float(np.mean(results["lines"])),
                    "mean_steps": float(np.mean(results["steps"])),
                }
            )
        print(
            f"{name:>7}, Mean: {np.mean(results['score']):7.1f} points"
            f"({np.mean(results['lines']):.1f} lines, {np.mean(results['steps']):.1f} pieces)"
        )


if __name__ == "__main__":
    main()
