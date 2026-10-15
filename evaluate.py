"""
uv run python evaluate.py
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db      # http://127.0.0.1:5000
"""

import mlflow
import numpy as np

from agents.baselines import GreedyAgent, RandomAgent
from tenten.env import TenTenEnv


def play_games(agent, env, n_games, seed=0):
    scores, lines, steps = [], [], []

    for i in range(n_games):
        obs, info = env.reset(seed=seed + i)
        done = False

        while not done:
            action = agent.act(env, info)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        scores.append(info["score"])
        lines.append(info["lines"])
        steps.append(info["steps"])

    return {"score": scores, "lines": lines, "steps": steps}


def main():
    n_games = 100
    env = TenTenEnv()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("baselines")

    for name, agent in [
        ("random", RandomAgent(seed=0)),
        ("greedy", GreedyAgent(seed=0)),
    ]:
        results = play_games(agent, env, n_games)
        with mlflow.start_run(run_name=name):
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
