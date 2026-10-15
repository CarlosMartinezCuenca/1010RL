"""Entrena la red de n-tuplas (TD(0) sobre afterstates) y lo guarda en MLflow.

Uso:
    uv run python train_ntuple.py
    uv run mlflow ui --backend-store-uri sqlite:///mlflow.db      # http://127.0.0.1:5000
"""

import os

import mlflow
import numpy as np

from agents.ntuple import NTupleAgent, SearchAgent
from evaluate import play_games
from tenten.env import TenTenEnv

CONFIG = {
    "total_steps": 4_500_000,
    # alpha va bajando por fases: primero aprende rápido, luego afina
    "alpha_phases": [(0, 0.1), (2_000_000, 0.025), (3_500_000, 0.006)],
    "gamma": 0.95,
    "reward_scale": 0.1,  # recompensa = puntos de la jugada × 0.1, como la DQN
    "loss_penalty": 0.0,  # sin castigo extra al perder (ponlo negativo para experimentar)
    "eps_start": 0.05,  # muy poca exploración: una jugada al azar suele matar
    "eps_decay_steps": 50_000,  # y a partir de aquí, ninguna
    "eval_every": 100_000,
    "eval_games": 30,
    "seed": 0,
}


def alpha_at(step, c):
    alpha = c["alpha_phases"][0][1]
    for start, value in c["alpha_phases"]:
        if step >= start:
            alpha = value
    return alpha


def epsilon_at(step, c):
    return c["eps_start"] * max(0.0, 1 - step / c["eps_decay_steps"])


def main():
    c = CONFIG
    env = TenTenEnv()
    eval_env = TenTenEnv()
    agent = NTupleAgent(
        gamma=c["gamma"],
        reward_scale=c["reward_scale"],
        loss_penalty=c["loss_penalty"],
        seed=c["seed"],
    )
    os.makedirs("models", exist_ok=True)
    best_score = -1
    errors = []

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("ntuple")
    with mlflow.start_run(run_name="ntuple"):
        mlflow.log_params({**c, "n_weights": len(agent.net.weights)})

        obs, info = env.reset(seed=c["seed"])
        previous = None
        for step in range(1, c["total_steps"] + 1):
            agent.alpha = alpha_at(step, c)

            action, afterstate, reward = agent.choose(env, epsilon_at(step, c))

            if previous is not None:
                target = reward + agent.gamma * agent.value(afterstate)
                errors.append(abs(agent.learn(previous, target)))

            obs, _, terminated, truncated, info = env.step(action)

            if terminated:
                errors.append(abs(agent.learn(afterstate, c["loss_penalty"])))
            previous = afterstate
            if terminated or truncated:
                previous = None
                obs, info = env.reset()

            if step % c["eval_every"] == 0:
                results = play_games(agent, eval_env, c["eval_games"], seed=10_000)
                score = float(np.mean(results["score"]))
                mlflow.log_metrics(
                    {
                        "eval_score": score,
                        "eval_lines": float(np.mean(results["lines"])),
                        "eval_steps": float(np.mean(results["steps"])),
                        "td_error": float(np.mean(errors)) if errors else 0.0,
                        "alpha": agent.alpha,
                    },
                    step=step,
                )
                errors = []
                print(
                    f"paso {step:>8} | puntos {score:8.1f} | alpha {agent.alpha}",
                    flush=True,
                )
                agent.save("models/ntuple_last.npy")
                if score > best_score:
                    best_score = score
                    agent.save("models/ntuple_best.npy")

        for name, player in [("final", agent), ("final_search", SearchAgent(agent))]:
            results = play_games(player, eval_env, c["eval_games"], seed=10_000)
            mlflow.log_metric(f"{name}_score", float(np.mean(results["score"])))
        mlflow.log_artifact("models/ntuple_last.npy")
        mlflow.log_artifact("models/ntuple_best.npy")
        print(f"Mejor evaluación: {best_score:.1f} puntos")


if __name__ == "__main__":
    main()
