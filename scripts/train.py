"""
Entrena un agente con su configuración por defecto (CONFIG en tenten/training/<agente>.py).

uv run python scripts/train.py --agent ntuple
uv run python scripts/train.py --agent dqn
uv run python scripts/train.py --agent tabular
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db      # http://127.0.0.1:5000
"""

import argparse

from tenten.evaluation import log_baselines
from tenten.game.env import TenTenEnv
from tenten.training import dqn, ntuple, tabular


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["tabular", "dqn", "ntuple"], required=True)
    args = parser.parse_args()

    if args.agent == "dqn":
        dqn.train(dqn.CONFIG, TenTenEnv, "dqn")
    elif args.agent == "ntuple":
        ntuple.train(ntuple.CONFIG, TenTenEnv, "ntuple")
    else:
        c = tabular.CONFIG
        experiment = f"tabular_{tabular.tag(c)}"
        log_baselines(c, lambda: tabular.make_env(c), experiment)
        tabular.train(c, lambda: tabular.make_env(c), experiment)


if __name__ == "__main__":
    main()
