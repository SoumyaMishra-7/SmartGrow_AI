from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TrainingConfig:
    scenario_name: str = "balanced"
    episodes: int = 160
    max_days: int = 30
    learning_rate: float = 0.18
    discount: float = 0.94
    epsilon: float = 1.0
    epsilon_decay: float = 0.985
    epsilon_min: float = 0.05
    seed: int = 7
    log_interval: int = 10
    save_plot: bool = True
