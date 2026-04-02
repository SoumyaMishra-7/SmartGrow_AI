from __future__ import annotations

from typing import Protocol


class Agent(Protocol):
    def act(self, observation: tuple[int, ...], training: bool = True) -> int:
        ...

    def learn(
        self,
        observation: tuple[int, ...],
        action: int,
        reward: float,
        next_observation: tuple[int, ...],
        terminated: bool,
    ) -> None:
        ...
