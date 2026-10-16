from tenten.agents.baselines import GreedyAgent, HeuristicAgent, RandomAgent
from tenten.agents.dqn import DQNAgent
from tenten.agents.ntuple import NTupleAgent, SearchAgent
from tenten.agents.tabular import TabularQAgent

NAMES = {
    "random": "Aleatorio",
    "greedy": "Voraz",
    "heuristic": "Heurística",
    "tabular": "Q-learning tabular",
    "dqn": "DQN",
    "ntuple": "N-tuplas",
    "ntuple_search": "N-tuplas + búsqueda",
}


def load_agent(name, env, model=None, hidden=256, seed=0):
    """Crea el agente `name` para `env` y, si aprende, carga sus pesos de `model`."""
    if name == "random":
        return RandomAgent(seed=seed)
    if name == "greedy":
        return GreedyAgent(seed=seed)
    if name == "heuristic":
        return HeuristicAgent(seed=seed)

    if name == "tabular":
        agent = TabularQAgent(env.action_space.n, seed=seed)
    elif name == "dqn":
        obs_size = env.observation_space.shape[0]
        agent = DQNAgent(obs_size, env.action_space.n, hidden=hidden, buffer_size=1, seed=seed)
    elif name in ("ntuple", "ntuple_search"):
        agent = NTupleAgent(size=env.size, seed=seed)
    else:
        raise ValueError(f"Agente desconocido: {name}")

    agent.load(model)
    return SearchAgent(agent) if name == "ntuple_search" else agent
