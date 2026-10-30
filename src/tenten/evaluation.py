"""Playing evaluation games and logging the baselines."""

import mlflow
import numpy as np

from tenten.agents.baselines import GreedyAgent, HeuristicAgent, RandomAgent
from tenten.tracking import setup_mlflow


def play_games(agent, env, n_games, seed=0):
    """Plays n_games greedy games (seeds seed, seed+1, ...); returns score, lines and steps lists."""
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


def log_baselines(c, make_env, experiment, seed=10_000):
    """Plays the three baselines on make_env() and logs one MLflow run each.
    Returns the per-game results of each baseline."""
    env = make_env()
    games = {}
    setup_mlflow(experiment)
    for name, baseline in [
        ("random", RandomAgent(seed=c["seed"])),
        ("greedy", GreedyAgent(seed=c["seed"])),
        ("heuristic", HeuristicAgent(seed=c["seed"])),
    ]:
        with mlflow.start_run(run_name=f"{name}_seed{c['seed']}"):
            mlflow.log_params({"agent": name, **c})
            games[name] = play_games(baseline, env, c["baseline_games"], seed=seed)
            means = {f"eval_{k}": float(np.mean(v)) for k, v in games[name].items()}
            mlflow.log_metrics(means)
    return games
