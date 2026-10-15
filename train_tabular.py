"""
uv run python train_tabular.py
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db      # http://127.0.0.1:5000
"""

import os

import mlflow
import numpy as np

from agents.baselines import GreedyAgent, HeuristicAgent, RandomAgent
from agents.tabular import TabularQAgent
from evaluate import play_games
from tenten.env import TenTenEnv

CONFIG = {
    "size": 5,
    "pieces": ["small_L", "small_L_c"],
    "max_steps": 200,
    "total_steps": 2_000_000,
    "alpha": 0.1,
    "gamma": 0.99,
    "eps_start": 1.0,
    "eps_end": 0.05,
    "eps_decay_steps": 1_000_000,
    "eval_every": 50_000,
    "eval_games": 20,
    "seed": 0,
}


def make_env(c):
    return TenTenEnv(
        size=c["size"], piece_names=c["pieces"], hand_size=1, max_steps=c["max_steps"]
    )


def epsilon_at(step, c):
    frac = min(1.0, step / c["eps_decay_steps"])
    return c["eps_start"] + frac * (c["eps_end"] - c["eps_start"])


def main():
    c = CONFIG
    env = make_env(c)
    eval_env = make_env(c)
    agent = TabularQAgent(
        env.action_space.n, alpha=c["alpha"], gamma=c["gamma"], seed=c["seed"]
    )
    os.makedirs("models", exist_ok=True)
    tag = "+".join(c["pieces"])

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("tabular_5x5")

    for name, baseline in [
        ("random", RandomAgent(seed=0)),
        ("greedy", GreedyAgent(seed=0)),
        ("heuristic", HeuristicAgent(seed=0)),
    ]:
        with mlflow.start_run(run_name=f"{name}_{tag}"):
            mlflow.log_params({"agent": name, **c})
            results = play_games(baseline, eval_env, c["eval_games"], seed=10_000)
            mlflow.log_metric("eval_lines", float(np.mean(results["lines"])))

    with mlflow.start_run(run_name=f"tabular_{tag}"):
        mlflow.log_params({"agent": "tabular", **c})

        obs, info = env.reset(seed=c["seed"])
        for step in range(1, c["total_steps"] + 1):
            key = agent.key(env)
            eps = epsilon_at(step, c)
            action = agent.choose(key, info["action_mask"], eps)
            obs, reward, terminated, truncated, info = env.step(action)

            agent.learn(
                key, action, reward, agent.key(env), info["action_mask"], terminated
            )
            if terminated or truncated:
                obs, info = env.reset()

            if step % c["eval_every"] == 0:
                results = play_games(agent, eval_env, c["eval_games"], seed=10_000)
                lines = float(np.mean(results["lines"]))
                mlflow.log_metrics(
                    {
                        "eval_lines": lines,
                        "eval_steps": float(np.mean(results["steps"])),
                        "states_seen": len(agent.Q),
                        "epsilon": eps,
                    },
                    step=step,
                )
                print(
                    f"paso {step:>8} | líneas {lines:6.1f} | estados vistos {len(agent.Q):>7}",
                    flush=True,
                )

        model_path = f"models/tabular_{tag}.pkl"
        agent.save(model_path)
        mlflow.log_artifact(model_path)


if __name__ == "__main__":
    main()
