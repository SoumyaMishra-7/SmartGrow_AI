from __future__ import annotations

import random


def epsilon_greedy(q_values: list[float], epsilon: float, rng: random.Random) -> int:
    if rng.random() < epsilon:
        return rng.randrange(len(q_values))
    best_value = max(q_values)
    return q_values.index(best_value)
