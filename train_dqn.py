"""
uv run python train_dqn.py
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db      # http://127.0.0.1:5000
"""

import os

import mlflow
import numpy as np
import torch

from agents.dqn import DQNAgent
from evaluate import play_games
from tenten.env import TenTenEnv

CONFIG = {
    "total_steps": 300_000,
    "hidden": 256,
    "lr": 1e-3,
    "gamma": 0.99,
    "buffer_size": 100_000,
    "batch_size": 128,
    "learning_starts": 5_000,
    "train_freq": 4,
    "reward_scale": 0.1,
    "target_update": 1_000,
    "eps_start": 1.0,
    "eps_end": 0.05,
    "eps_decay_steps": 100_000,
    "eval_every": 10_000,
    "eval_games": 20,
    "seed": 0,
}


torch.set_num_threads(1)


def epsilon_at(step, c):
    frac = min(1.0, step / c["eps_decay_steps"])
    return c["eps_start"] + frac * (c["eps_end"] - c["eps_start"])


def main():
    c = CONFIG
    env = TenTenEnv()
    eval_env = TenTenEnv()
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = DQNAgent(
        obs_size,
        n_actions,
        hidden=c["hidden"],
        lr=c["lr"],
        gamma=c["gamma"],
        buffer_size=c["buffer_size"],
        batch_size=c["batch_size"],
        seed=c["seed"],
    )

    os.makedirs("models", exist_ok=True)
    best_score = -1
    losses = []

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("dqn")
    with mlflow.start_run(run_name="dqn"):
        mlflow.log_params(c)

        obs, info = env.reset(seed=c["seed"])
        for step in range(1, c["total_steps"] + 1):
            eps = epsilon_at(step, c)

            action = agent.choose(obs, info["action_mask"], eps)
            next_obs, reward, terminated, truncated, info = env.step(action)
            reward *= c["reward_scale"]

            agent.remember(
                obs, action, reward, next_obs, info["action_mask"], terminated
            )
            obs = next_obs
            if terminated or truncated:
                obs, info = env.reset()

            if step >= c["learning_starts"] and step % c["train_freq"] == 0:
                losses.append(agent.learn())

            if step % c["target_update"] == 0:
                agent.update_target()

            if step % c["eval_every"] == 0:
                results = play_games(agent, eval_env, c["eval_games"], seed=10_000)
                score = float(np.mean(results["score"]))
                mlflow.log_metrics(
                    {
                        "eval_score": score,
                        "eval_lines": float(np.mean(results["lines"])),
                        "eval_steps": float(np.mean(results["steps"])),
                        "epsilon": eps,
                        "loss": float(np.mean(losses)) if losses else 0.0,
                    },
                    step=step,
                )
                losses = []
                print(
                    f"paso {step:>7} | puntos {score:7.1f} | epsilon {eps:.2f}",
                    flush=True,
                )

                if score > best_score:
                    best_score = score
                    agent.save("models/dqn_best.pt")

        mlflow.log_artifact("models/dqn_best.pt")
        print(f"Mejor evaluación: {best_score:.1f} puntos")


if __name__ == "__main__":
    main()
