from __future__ import annotations

from env.state import GardenState
from utils.helpers import clamp


def calculate_reward(previous: GardenState, current: GardenState, terminated: bool) -> float:
    growth_reward = (current.growth - previous.growth) * 12.0
    health_reward = (current.health - previous.health) * 8.0
    stability_reward = (current.health + current.soil_moisture + current.nutrients) * 0.2
    stress_penalty = (
        abs(current.soil_moisture - 0.62) * 0.8
        + abs(current.nutrients - 0.58) * 0.7
        + clamp((current.temperature - 24.0) / 12.0, -1.0, 1.0) ** 2 * 0.4
    ) * -1.0
    resource_penalty = ((1.0 - current.water_tank) + (1.0 - current.nutrient_tank) + (1.0 - current.energy_reserve)) * -0.2
    milestone_reward = 0.0
    if previous.growth < 0.35 <= current.growth:
        milestone_reward += 1.0
    if previous.growth < 0.75 <= current.growth:
        milestone_reward += 1.5
    if previous.health < 0.50 <= current.health:
        milestone_reward += 0.5
    terminal_bonus = 18.0 if terminated and current.growth >= 1.0 and current.health >= 0.65 else 0.0
    failure_penalty = -20.0 if current.health <= 0.15 else 0.0
    return growth_reward + health_reward + stability_reward + stress_penalty + resource_penalty + milestone_reward + terminal_bonus + failure_penalty
