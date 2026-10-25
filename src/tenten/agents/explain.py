"""
Explanations for the n-tuple agent.

V(board) is a plain sum of one weight per tuple, so it can be split EXACTLY into
the contribution of each cell (no approximation, unlike saliency maps for deep nets).
"""

import numpy as np


def cell_contributions(net, board):
    """
    Splits V(board) into one number per cell; the numbers add up exactly to V(board).

    Each tuple's active weight is shared equally among the cells of that tuple.
    Positive cells raise the value of the board, negative cells lower it.
    """
    weights = net.weights[net.indices(board[None, :])[0]]  # active weight of each tuple
    members = net.powers > 0  # members[cell, t] = True if the cell belongs to tuple t
    share = weights / members.sum(
        axis=0
    )  # part of each weight that goes to each of its cells
    return members @ share


def compare_moves(agent, env):
    """
    Explains why the agent prefers its move over the runner-up (second best move).

    score = reward + gamma * V(afterstate), so the score gap between the two moves is
        (reward_best - reward_second) + sum(diff_map)
    where diff_map[cell] = gamma * (contribution_best - contribution_second).
    Returns None if every legal move leads to the same board.
    """
    actions, _, afterstates, rewards, scores = agent.candidates(env)
    order = np.argsort(-scores)
    best = order[0]

    # Runner-up: the best move that leads to a different board
    others = [
        i for i in order[1:] if not np.array_equal(afterstates[i], afterstates[best])
    ]
    if not others:
        return None
    second = others[0]

    def move(i):
        value = agent.value(afterstates[i])
        return {
            "action": int(actions[i]),
            "afterstate": afterstates[i],
            "reward": float(rewards[i]),
            "value": value,
            "score": float(scores[i]),
            # candidates() replaces V by loss_penalty when no piece left in hand fits
            "certain_loss": not np.isclose(scores[i], rewards[i] + agent.gamma * value),
        }

    b, s = move(best), move(second)
    diff_map = agent.gamma * (
        cell_contributions(agent.net, b["afterstate"])
        - cell_contributions(agent.net, s["afterstate"])
    )
    return {"best": b, "second": s, "diff_map": diff_map}
