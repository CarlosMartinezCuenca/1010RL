# 1010! Reinforcement Learning

Reinforcement learning agents that learn to play **1010!**: place pieces on a 10×10 board, clear full rows and columns, and survive as long as possible. Everything runs on CPU.

<p align="center">
  <img src="results/10x10_all/best_ntuple_search.gif" alt="N-tuple network with one-step search playing 1010!" width="600">
</p>

<p align="center"><em>N-tuple network + one-step search playing the full game.</em></p>

## Agents

| Agent                     | Idea                                                                                           |
| ------------------------- | ---------------------------------------------------------------------------------------------- |
| Random, Greedy, Heuristic | Baselines: random move, most lines cleared, hand-made board score                              |
| Tabular Q-learning        | One Q-value per visited (state, action); only feasible on small boards                         |
| DQN                       | Double DQN with action masking, replay buffer and target network                               |
| N-tuples                  | TD(0) on afterstates with an n-tuple network that shares weights across the 8 board symmetries |
| N-tuples + search         | The trained n-tuple network plus a one-step lookahead using the pieces left in hand            |

All agents use the same reward: the points of each move (cells placed + line bonus).

## Results

Full game: 10×10 board, all 19 pieces, hand of 3, at most 1000 pieces per game. DQN and n-tuples get the same budget of 1M training steps; every agent is evaluated on the same 50 games.

<p align="center">
  <img src="results/10x10_all/final_comparison.png" alt="Mean score of each agent on the 10x10 game" width="700">
</p>

| Agent                 | Mean score |
| --------------------- | ---------- |
| **N-tuples + search** | **8542**   |
| N-tuples              | 3418       |
| Heuristic             | 1109       |
| Greedy                | 120        |
| DQN                   | 98         |
| Random                | 79         |

The n-tuple network generalises across boards it has never seen, while the DQN does not learn in this setting. On a small deterministic version (5×5, one piece) the ranking flips and the DQN wins, since it can memorise a sequence that never loses. All figures are in [`results/`](results/).

## Project structure

```
src/tenten/
  game/        rules engine and Gymnasium environment
  agents/      baselines, tabular, DQN, n-tuples, and a registry to load any of them
  training/    one training loop per agent, with its default CONFIG
  evaluation.py, tracking.py (MLflow), plots.py
scripts/       command-line entry points
tests/         mirrors src/tenten
results/       experiment figures, GIFs and summary.json
```

## Usage

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync

# Compare all agents (figures go to results/<setup>/)
uv run python scripts/experiment.py --setup 10x10_all
uv run python scripts/experiment.py --setup 5x5_small_L

# Train a single agent with its default configuration
uv run python scripts/train.py --agent ntuple      # tabular | dqn | ntuple

# Watch an agent play (or save a GIF with --gif file.gif)
uv run python scripts/visualize.py --agent ntuple --search

# Experiment tracking and tests
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
uv run pytest
```

## Future work

- **Explainability.** The n-tuple value is a sum of table weights, so it can be decomposed exactly: per-cell heatmaps of what the agent considers good or bad, contrastive explanations of why one move beats another, and inspection of the patterns each table has learned.
- **Convolutional DQN.** Replace the MLP with a CNN over the board, as in the Atari DQN, so the network can exploit the spatial structure of the board.

## References

- Mnih, V. et al. (2015). _Human-level control through deep reinforcement learning_. Nature, 518, 529–533.
- van Hasselt, H., Guez, A., & Silver, D. (2016). _Deep Reinforcement Learning with Double Q-learning_. AAAI.
- Sutton, R. S., & Barto, A. G. (2018). _Reinforcement Learning: An Introduction_ (2nd ed.). MIT Press.
- Lucas, S. M. (2008). _Learning to Play Othello with N-Tuple Systems_. Australian Journal of Intelligent Information Processing Systems.
- Szubert, M., & Jaśkowski, W. (2014). _Temporal Difference Learning of N-Tuple Networks for the Game 2048_. IEEE Conference on Computational Intelligence and Games (CIG).
