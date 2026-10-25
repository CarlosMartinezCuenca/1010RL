"""Tabular Q-learning training loop (only feasible on small boards)."""

import os

import mlflow
import numpy as np

from tenten.agents.tabular import TabularQAgent
from tenten.evaluation import play_games
from tenten.game.env import TenTenEnv
from tenten.tracking import setup_mlflow

CONFIG = {  # 5x5 board with two L pieces that tile a 2x3 rectangle
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
    "baseline_games": 20,
    "seed": 0,
}


def make_env(c):
    """Small environment described by the config (hand of one piece)."""
    return TenTenEnv(
        size=c["size"], piece_names=c["pieces"], hand_size=1, max_steps=c["max_steps"]
    )


def tag(c):
    """Short name of the environment, e.g. 5x5_small_L+small_L_c."""
    return f"{c['size']}x{c['size']}_{'+'.join(c['pieces'])}"


def epsilon_at(step, c):
    """Linear decay from eps_start to eps_end, then constant."""
    frac = min(1.0, step / c["eps_decay_steps"])
    return c["eps_start"] + frac * (c["eps_end"] - c["eps_start"])


def train(c, make_env, experiment):
    """Trains on make_env() and logs to MLflow experiment `experiment`.
    Returns the evaluation history and the path of the best checkpoint."""
    env = make_env()
    eval_env = make_env()
    agent = TabularQAgent(
        env.action_space.n, alpha=c["alpha"], gamma=c["gamma"], seed=c["seed"]
    )
    run_name = f"tabular_seed{c['seed']}"
    model_path = f"models/{experiment}/{run_name}_best.pkl"
    os.makedirs(f"models/{experiment}", exist_ok=True)
    best_score = -1
    history = []

    setup_mlflow(experiment)
    with mlflow.start_run(run_name=run_name):
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
                score = float(np.mean(results["score"]))
                lines = float(np.mean(results["lines"]))
                mlflow.log_metrics(
                    {
                        "eval_score": score,
                        "eval_lines": lines,
                        "eval_steps": float(np.mean(results["steps"])),
                        "states_seen": len(agent.Q),
                        "epsilon": eps,
                    },
                    step=step,
                )
                history.append(
                    {"step": step, **{k: float(np.mean(v)) for k, v in results.items()}}
                )
                print(
                    f"step {step:>8} | score {score:7.1f} | states seen {len(agent.Q):>7}",
                    flush=True,
                )
                if score > best_score:
                    best_score = score
                    agent.save(model_path)

        mlflow.log_artifact(model_path)
        print(f"Best evaluation: {best_score:.1f} points")
    return history, model_path
