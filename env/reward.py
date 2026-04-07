from __future__ import annotations

from env.state import GardenState
from utils.helpers import clamp


def calculate_reward(
    previous: GardenState,
    current: GardenState,
    terminated: bool,
    usage: dict[str, float] | None = None,
) -> float:
    usage = usage or {"water_ml": 0.0, "nutrient_ml": 0.0, "energy_wh": 0.0}

    growth_progress = max(0.0, current.growth - previous.growth) * 10.0
    health_progress = (current.health - previous.health) * 7.0

    target_moisture = 0.60
    target_nutrients = 0.55
    climate_stability = 1.0 - (
        abs(current.soil_moisture - target_moisture) * 0.7
        + abs(current.nutrients - target_nutrients) * 0.6
        + clamp((current.temperature - 24.0) / 14.0, -1.0, 1.0) ** 2 * 0.5
    )

    water_penalty = max(0.0, usage["water_ml"] - 220.0) / 220.0
    nutrient_penalty = max(0.0, usage["nutrient_ml"] - 80.0) / 80.0
    energy_penalty = max(0.0, usage["energy_wh"] - 120.0) / 120.0
    efficiency_penalty = (water_penalty + nutrient_penalty + energy_penalty) * 1.4

    reserve_guard_bonus = (current.water_tank + current.energy_reserve + current.nutrient_tank) * 0.15
    damage_penalty = -8.0 if current.health <= 0.20 else 0.0
    terminal_success = 12.0 if terminated and current.growth >= 1.0 and current.health >= 0.70 else 0.0

    return growth_progress + health_progress + climate_stability + reserve_guard_bonus - efficiency_penalty + damage_penalty + terminal_success
