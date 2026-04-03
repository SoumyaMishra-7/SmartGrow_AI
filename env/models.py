from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(slots=True)
class ActionEffect:
    label: str = "observe"
    water_delta: float = 0.0
    nutrient_delta: float = 0.0
    shade_delta: float = 0.0
    fan_delta: float = 0.0
    energy_delta: float = 0.0


@dataclass(slots=True)
class StepResult:
    observation: tuple[int, ...]
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, float | int | str]


@dataclass(slots=True)
class EpisodeTrace:
    rewards: List[float] = field(default_factory=list)
    harvest_scores: List[float] = field(default_factory=list)
    stress_scores: List[float] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class OpenEnvActionModel:
    id: int
    name: str
    description: str


@dataclass(frozen=True, slots=True)
class OpenEnvStateModel:
    day: int
    soil_moisture: float
    nutrients: float
    temperature: float
    humidity: float
    light: float
    growth: float
    health: float
    water_tank: float
    nutrient_tank: float
    energy_reserve: float
    last_action: str


@dataclass(frozen=True, slots=True)
class OpenEnvResetModel:
    observation: tuple[int, ...]
    info: Dict[str, str | int | float]
    state: OpenEnvStateModel


@dataclass(frozen=True, slots=True)
class OpenEnvStepModel:
    observation: tuple[int, ...]
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, str | int | float]
    state: OpenEnvStateModel
