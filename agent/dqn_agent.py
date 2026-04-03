from __future__ import annotations

import random
from collections import defaultdict

from agent.policy import epsilon_greedy
from agent.replay_buffer import ReplayBuffer, Transition


class DQNAgent:
    """
    Lightweight tabular agent kept under the DQN name so the project structure
    stays aligned with the original intent while remaining dependency-light.
    """

    def __init__(
        self,
        action_size: int,
        learning_rate: float = 0.18,
        discount: float = 0.94,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.985,
        epsilon_min: float = 0.05,
        seed: int = 7,
    ) -> None:
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.rng = random.Random(seed)
        self.q_table: defaultdict[tuple[int, ...], list[float]] = defaultdict(lambda: [0.0] * action_size)
        self.replay_buffer = ReplayBuffer()

    def act(self, observation: tuple[int, ...], training: bool = True) -> int:
        q_values = self.q_table[observation]
        epsilon = self.epsilon if training else 0.0
        return epsilon_greedy(q_values, epsilon, self.rng)

    def learn(
        self,
        observation: tuple[int, ...],
        action: int,
        reward: float,
        next_observation: tuple[int, ...],
        terminated: bool,
    ) -> float:
        self.replay_buffer.add(Transition(observation, action, reward, next_observation, terminated))
        current_q = self.q_table[observation][action]
        next_best = 0.0 if terminated else max(self.q_table[next_observation])
        target = reward + self.discount * next_best
        td_error = target - current_q
        self.q_table[observation][action] = current_q + self.learning_rate * td_error
        return td_error * td_error

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
