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


@dataclass(frozen=True, slots=True)
class ManagementAction:
    water_ml: int = 0
    sunlight_adjustment: str = "maintain"
    nutrients: str = "hold"
    energy_mode: str = "normal"


@dataclass(slots=True)
class StepResult:
    observation: tuple[int, ...] | Dict[str, float | int | str]
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, float | int | str | bool]


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
    observation: tuple[int, ...] | Dict[str, float | int | str]
    info: Dict[str, str | int | float | bool]
    state: OpenEnvStateModel


@dataclass(frozen=True, slots=True)
class OpenEnvStepModel:
    observation: tuple[int, ...] | Dict[str, float | int | str]
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, str | int | float | bool | dict]
    state: OpenEnvStateModel
