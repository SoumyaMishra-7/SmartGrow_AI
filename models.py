from __future__ import annotations

from typing import Literal

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
