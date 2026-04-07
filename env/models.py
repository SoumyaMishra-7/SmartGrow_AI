from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plant_health: float = Field(..., ge=0.0, le=1.0)
    soil_moisture: float = Field(..., ge=0.0, le=1.0)
    weather: str
    water_remaining: float = Field(..., ge=0.0)
    energy_budget: float = Field(..., ge=0.0)
    step: int = Field(..., ge=0)


class Action(BaseModel):
    model_config = ConfigDict(extra="forbid")

    water_ml: float = Field(0.0, ge=0.0, le=400.0)
    sunlight_adjustment: Literal["increase", "decrease", "maintain"] = "maintain"
    nutrients: Literal["add", "none"] = "none"
    energy_mode: Literal["low", "normal", "high"] = "normal"


class Reward(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: float = Field(..., ge=0.0, le=1.0)
    reason: str


class PlantSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plant_id: int = Field(..., ge=1)
    health: float = Field(..., ge=0.0, le=1.0)
    growth: float = Field(..., ge=0.0, le=1.5)


class InternalState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_type: Literal["easy", "medium", "hard"]
    scenario_name: str
    plant_count: int = Field(..., ge=1)
    seed: int
    step_count: int = Field(..., ge=0)
    max_steps: int = Field(..., ge=1)
    weather: str
    soil_moisture: float = Field(..., ge=0.0, le=1.0)
    nutrients: float = Field(..., ge=0.0, le=1.0)
    temperature: float = Field(..., ge=0.0, le=45.0)
    humidity: float = Field(..., ge=0.0, le=1.0)
    light: float = Field(..., ge=0.0, le=1.0)
    water_remaining: float = Field(..., ge=0.0)
    nutrient_remaining: float = Field(..., ge=0.0)
    energy_budget: float = Field(..., ge=0.0)
    average_growth: float = Field(..., ge=0.0, le=1.5)
    average_health: float = Field(..., ge=0.0, le=1.0)
    plants: list[PlantSnapshot]
    last_action: Action | None = None
    last_reward: Reward | None = None
    terminated: bool = False


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
