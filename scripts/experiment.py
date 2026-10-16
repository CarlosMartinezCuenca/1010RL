"""
Compara los agentes que aprenden con las baselines, con el mismo presupuesto de pasos.

uv run python scripts/experiment.py --setup 5x5_small_L
uv run python scripts/experiment.py --setup 10x10_all
uv run python scripts/experiment.py --setup 10x10_all --only ntuple    # reentrena solo ese modelo
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db      # http://127.0.0.1:5000
"""

import argparse
import json
import os

import mlflow
import numpy as np

from tenten.agents.registry import NAMES, load_agent
from tenten.evaluation import log_baselines, play_games
from tenten.game.env import TenTenEnv
from tenten.game.pieces import PIECES
from tenten.plots import plot_curves, plot_final
from tenten.tracking import delete_runs, setup_mlflow
from tenten.training import dqn as train_dqn
from tenten.training import ntuple as train_ntuple
from tenten.training import tabular as train_tabular

SEED = 0
FINAL_SEED = 20_000  # todas las evaluaciones finales juegan las mismas partidas
BASELINES = ("heuristic", "greedy", "random")
TRAINERS = {"tabular": train_tabular, "dqn": train_dqn, "ntuple": train_ntuple}


def phases(total):
    """Las fases de alpha de las n-tuplas, escaladas al presupuesto."""
    return [(0, 0.1), (total * 4 // 9, 0.025), (total * 7 // 9, 0.006)]


SETUPS = {
    # Una sola pieza: el juego es determinista, así que todos necesitan explorar bastante
    "5x5_small_L": {
        "env": {
            "size": 5,
            "piece_names": ["small_L"],
            "hand_size": 1,
            "max_steps": 200,
        },
        "total_steps": 1_000_000,
        "eval_every": 20_000,
        "eval_games": 5,
        "final_games": 20,
        "configs": {
            "tabular": {
                "size": 5,
                "pieces": ["small_L"],
                "max_steps": 200,
                "eps_decay_steps": 500_000,
            },
            "dqn": {"eps_decay_steps": 333_333},
            "ntuple": {
                "alpha_phases": phases(1_000_000),
                "eps_start": 1.0,
                "eps_end": 0.05,
                "eps_decay_steps": 333_333,
            },
        },
        "extra": [],
    },
    # El juego completo: el azar de las piezas ya obliga a ver tableros nuevos
    "10x10_all": {
        "env": {
            "size": 10,
            "piece_names": list(PIECES),
            "hand_size": 3,
            "max_steps": 1000,
        },
        "total_steps": 1_000_000,
        "eval_every": 50_000,
        "eval_games": 10,
        "final_games": 50,
        "configs": {
            "dqn": {"eps_decay_steps": 333_333},
            "ntuple": {"alpha_phases": phases(1_000_000)},
        },
        "extra": ["ntuple_search"],  # n-tuplas + búsqueda, solo en la evaluación final
    },
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", choices=list(SETUPS), default="5x5_small_L")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=list(TRAINERS),
        help="reentrena solo estos modelos y reutiliza el resto de summary.json",
    )
    args = parser.parse_args()

    setup = SETUPS[args.setup]
    results_dir = f"results/{args.setup}"
    os.makedirs(results_dir, exist_ok=True)

    def make_env():
        return TenTenEnv(**setup["env"])

    common = {
        "total_steps": setup["total_steps"],
        "eval_every": setup["eval_every"],
        "eval_games": setup["eval_games"],
        "seed": SEED,
    }
    configs = {
        name: {**TRAINERS[name].CONFIG, **common, **overrides}
        for name, overrides in setup["configs"].items()
    }

    if args.only:
        with open(f"{results_dir}/summary.json", encoding="utf-8") as f:
            previous = json.load(f)
        baselines = {n: previous["final_scores"][n] for n in BASELINES}
        histories, models = previous["histories"], previous["models"]
        delete_runs(
            args.setup, [f"{n}_seed{SEED}" for n in args.only] + [f"summary_seed{SEED}"]
        )
    else:
        baseline_config = {
            **common,
            "baseline_games": setup["final_games"],
            **setup["env"],
        }
        baselines = log_baselines(
            baseline_config, make_env, args.setup, seed=FINAL_SEED
        )
        histories, models = {}, {}

    for name in args.only or configs:
        print(f"\n=== {NAMES[name]} ===", flush=True)
        histories[name], models[name] = TRAINERS[name].train(
            configs[name], make_env, args.setup
        )

    # Evaluación final: el mejor modelo de cada agente en las mismas partidas que las baselines
    env = make_env()
    final = dict(baselines)
    capped = {}  # % de partidas que llegan a max_steps (el tope, no una derrota)
    players = {**models, **{extra: models["ntuple"] for extra in setup["extra"]}}
    for name, path in players.items():
        print(f"Evaluación final: {NAMES[name]}", flush=True)
        agent = load_agent(
            name, env, path, hidden=train_dqn.CONFIG["hidden"], seed=SEED
        )
        results = play_games(agent, env, setup["final_games"], seed=FINAL_SEED)
        final[name] = float(np.mean(results["score"]))
        capped[name] = 100 * float(np.mean(np.array(results["steps"]) >= env.max_steps))
    best = max(players, key=final.get)

    references = {NAMES[n]: baselines[n] for n in BASELINES}
    total = setup["total_steps"]
    plot_curves(
        histories,
        "score",
        "Puntuación durante el entrenamiento",
        "Puntos por partida",
        f"{results_dir}/learning_curves.png",
        references,
        total,
    )
    plot_curves(
        histories,
        "steps",
        "Supervivencia durante el entrenamiento",
        "Piezas colocadas",
        f"{results_dir}/survival.png",
        {"Máximo": setup["env"]["max_steps"]},
        total,
    )
    plot_final(final, setup["final_games"], f"{results_dir}/final_comparison.png")

    summary = {
        "setup": args.setup,
        "final_scores": final,
        "capped_pct": capped,
        "best": best,
        "best_model": players[best],
        "models": models,
        "histories": histories,
    }
    with open(f"{results_dir}/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    setup_mlflow(args.setup)
    with mlflow.start_run(run_name=f"summary_seed{SEED}"):
        mlflow.log_metrics({f"final_score_{n}": s for n, s in final.items()})
        mlflow.log_artifacts(results_dir)

    print("\nPuntuación final:")
    for name in sorted(final, key=final.get, reverse=True):
        tope = f"  ({capped[name]:.0f}% llegan al tope)" if name in capped else ""
        print(f"  {NAMES[name]:<20} {final[name]:7.1f}{tope}")
    print(f"Mejor: {NAMES[best]} -> {players[best]}")


if __name__ == "__main__":
    main()
