from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(slots=True)
class Transition:
    observation: tuple[int, ...]
    action: int
    reward: float
    next_observation: tuple[int, ...]
    terminated: bool


class ReplayBuffer:
    def __init__(self, capacity: int = 5000) -> None:
        self.capacity = capacity
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def add(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def __len__(self) -> int:
        return len(self.buffer)
