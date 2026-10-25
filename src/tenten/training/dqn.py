"""DQN training loop."""

import os

import mlflow
import numpy as np
import torch

from tenten.agents.dqn import DQNAgent
from tenten.evaluation import play_games
from tenten.tracking import setup_mlflow

CONFIG = {  # defaults for the full 10x10 game
    "total_steps": 300_000,
    "hidden": 256,
    "lr": 1e-3,
    "gamma": 0.99,
    "buffer_size": 100_000,
    "batch_size": 128,
    "learning_starts": 5_000,
    "train_freq": 4,
    "reward_scale": 0.1,  # keeps Q-values small for the network
    "target_update": 1_000,
    "eps_start": 1.0,
    "eps_end": 0.05,
    "eps_decay_steps": 100_000,
    "eval_every": 10_000,
    "eval_games": 20,
    "seed": 0,
}


def epsilon_at(step, c):
    """Linear decay from eps_start to eps_end, then constant."""
    frac = min(1.0, step / c["eps_decay_steps"])
    return c["eps_start"] + frac * (c["eps_end"] - c["eps_start"])


def train(c, make_env, experiment):
    """Trains on make_env() and logs to MLflow experiment `experiment`.
    Returns the evaluation history and the path of the best checkpoint."""
    torch.set_num_threads(1)  # the network is too small to benefit from more threads
    env = make_env()
    eval_env = make_env()
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

    run_name = f"dqn_seed{c['seed']}"
    model_path = f"models/{experiment}/{run_name}_best.pt"
    os.makedirs(f"models/{experiment}", exist_ok=True)
    best_score = -1
    losses = []
    history = []

    setup_mlflow(experiment)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(c)

        obs, info = env.reset(seed=c["seed"])
        for step in range(1, c["total_steps"] + 1):
            eps = epsilon_at(step, c)

            action = agent.choose(obs, info["action_mask"], eps)
            next_obs, reward, terminated, truncated, info = env.step(action)
            reward *= c["reward_scale"]

            # Only `terminated` is stored: a truncated game still has a future to bootstrap from
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
                history.append(
                    {"step": step, **{k: float(np.mean(v)) for k, v in results.items()}}
                )
                print(
                    f"step {step:>7} | score {score:7.1f} | epsilon {eps:.2f}",
                    flush=True,
                )

                if score > best_score:
                    best_score = score
                    agent.save(model_path)

        mlflow.log_artifact(model_path)
        print(f"Best evaluation: {best_score:.1f} points")
    return history, model_path
