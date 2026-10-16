import mlflow
import numpy as np

from tenten.agents.baselines import GreedyAgent, HeuristicAgent, RandomAgent
from tenten.tracking import setup_mlflow


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


def log_baselines(c, make_env, experiment, seed=10_000):
    """Las baselines en el mismo entorno, para poder comparar. Devuelve su puntuación media."""
    env = make_env()
    scores = {}
    setup_mlflow(experiment)
    for name, baseline in [
        ("random", RandomAgent(seed=c["seed"])),
        ("greedy", GreedyAgent(seed=c["seed"])),
        ("heuristic", HeuristicAgent(seed=c["seed"])),
    ]:
        with mlflow.start_run(run_name=f"{name}_seed{c['seed']}"):
            mlflow.log_params({"agent": name, **c})
            results = play_games(baseline, env, c["baseline_games"], seed=seed)
            scores[name] = float(np.mean(results["score"]))
            mlflow.log_metrics(
                {
                    "eval_score": scores[name],
                    "eval_lines": float(np.mean(results["lines"])),
                    "eval_steps": float(np.mean(results["steps"])),
                }
            )
    return scores
