from __future__ import annotations

from env.models import InternalState
from utils.helpers import clamp


def _normalized_growth(state: InternalState) -> float:
    return clamp(state.average_growth / 1.2, 0.0, 1.0)


def grade_easy(state: InternalState) -> float:
    score = (
        state.average_health * 0.50
        + _normalized_growth(state) * 0.25
        + clamp(state.water_remaining / 1000.0, 0.0, 1.0) * 0.15
        + clamp(state.energy_budget / 1000.0, 0.0, 1.0) * 0.10
    )
    return round(clamp(score, 0.0, 1.0), 4)


def grade_medium(state: InternalState) -> float:
    multi_plant_stability = sum(plant.health for plant in state.plants) / max(1, len(state.plants))
    score = (
        _normalized_growth(state) * 0.35
        + state.average_health * 0.30
        + clamp(multi_plant_stability, 0.0, 1.0) * 0.20
        + clamp((state.water_remaining + state.energy_budget) / 2000.0, 0.0, 1.0) * 0.15
    )
    return round(clamp(score, 0.0, 1.0), 4)


def grade_hard(state: InternalState) -> float:
    resource_efficiency = clamp(
        (state.water_remaining + state.nutrient_remaining + state.energy_budget) / 3000.0,
        0.0,
        1.0,
    )
    plant_resilience = sum(min(1.0, plant.health + (plant.growth / 1.5) * 0.5) for plant in state.plants) / max(1, len(state.plants))
    score = (
        _normalized_growth(state) * 0.30
        + state.average_health * 0.25
        + clamp(plant_resilience, 0.0, 1.0) * 0.25
        + resource_efficiency * 0.20
    )
    return round(clamp(score, 0.0, 1.0), 4)
