"""Playing evaluation games and logging the baselines."""

from concurrent.futures import ProcessPoolExecutor

import mlflow
import numpy as np

from tenten.agents.baselines import GreedyAgent, HeuristicAgent, RandomAgent
from tenten.tracking import setup_mlflow


def play_game(agent, env, seed):
    """Plays one greedy game; returns its score, lines and steps."""
    obs, info = env.reset(seed=seed)
    done = False

    while not done:
        action = agent.act(env, info)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    return info["score"], info["lines"], info["steps"]


def play_games(agent, env, n_games, seed=0, workers=1):
    """Plays n_games greedy games (seeds seed, seed+1, ...), on `workers` processes.
    Returns score, lines and steps lists."""
    seeds = list(range(seed, seed + n_games))

    if workers == 1:
        games = [play_game(agent, env, s) for s in seeds]
    else:
        # Each game is independent: every process gets its own copy of agent and env
        with ProcessPoolExecutor(workers) as pool:
            games = list(pool.map(play_game, [agent] * n_games, [env] * n_games, seeds))

    scores, lines, steps = (list(column) for column in zip(*games))
    return {"score": scores, "lines": lines, "steps": steps}


def log_baselines(c, make_env, experiment, seed=10_000, workers=1):
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
            games[name] = play_games(baseline, env, c["baseline_games"], seed, workers)
            means = {f"eval_{k}": float(np.mean(v)) for k, v in games[name].items()}
            mlflow.log_metrics(means)
    return games
