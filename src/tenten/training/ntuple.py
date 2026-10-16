"""N-tuple network training loop: TD(0) on afterstates."""

import os

import mlflow
import numpy as np

from tenten.agents.ntuple import NTupleAgent, SearchAgent
from tenten.evaluation import play_games
from tenten.tracking import setup_mlflow

CONFIG = {  # defaults for the full 10x10 game
    "total_steps": 4_500_000,
    "alpha_phases": [(0, 0.1), (2_000_000, 0.025), (3_500_000, 0.006)],  # (from step, alpha)
    "gamma": 0.95,
    "reward_scale": 0.1,
    "loss_penalty": 0.0,  # value of a lost game; < 0 is extra reward shaping
    "eps_start": 0.05,
    "eps_end": 0.0,
    "eps_decay_steps": 50_000,
    "eval_every": 100_000,
    "eval_games": 30,
    "seed": 0,
}


def alpha_at(step, c):
    """Learning rate of the current phase."""
    alpha = c["alpha_phases"][0][1]
    for start, value in c["alpha_phases"]:
        if step >= start:
            alpha = value
    return alpha


def epsilon_at(step, c):
    """Linear decay from eps_start to eps_end, then constant."""
    frac = max(0.0, 1 - step / c["eps_decay_steps"])
    return c["eps_end"] + frac * (c["eps_start"] - c["eps_end"])


def train(c, make_env, experiment):
    """Trains on make_env() and logs to MLflow experiment `experiment`.
    Returns the evaluation history and the path of the best checkpoint."""
    env = make_env()
    eval_env = make_env()
    agent = NTupleAgent(
        size=env.size,
        gamma=c["gamma"],
        reward_scale=c["reward_scale"],
        loss_penalty=c["loss_penalty"],
        seed=c["seed"],
    )
    run_name = f"ntuple_seed{c['seed']}"
    last_path = f"models/{experiment}/{run_name}_last.npy"
    best_path = f"models/{experiment}/{run_name}_best.npy"
    os.makedirs(f"models/{experiment}", exist_ok=True)
    best_score = -1
    errors = []
    history = []

    setup_mlflow(experiment)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({**c, "n_weights": len(agent.net.weights)})

        obs, info = env.reset(seed=c["seed"])
        previous = None
        for step in range(1, c["total_steps"] + 1):
            agent.alpha = alpha_at(step, c)

            action, afterstate, reward = agent.choose(env, epsilon_at(step, c))

            # The previous afterstate is worth this move's reward plus the value of where it leads
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
                history.append({"step": step, **{k: float(np.mean(v)) for k, v in results.items()}})
                print(
                    f"step {step:>8} | score {score:8.1f} | alpha {agent.alpha}",
                    flush=True,
                )
                agent.save(last_path)
                if score > best_score:
                    best_score = score
                    agent.save(best_path)

        for name, player in [("final", agent), ("final_search", SearchAgent(agent))]:
            results = play_games(player, eval_env, c["eval_games"], seed=10_000)
            mlflow.log_metric(f"{name}_score", float(np.mean(results["score"])))
        mlflow.log_artifact(last_path)
        mlflow.log_artifact(best_path)
        print(f"Best evaluation: {best_score:.1f} points")
    return history, best_path

