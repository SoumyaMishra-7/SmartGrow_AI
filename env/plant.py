from __future__ import annotations

from env.state import GardenState
from env.weather import WeatherState
from utils.helpers import clamp


def update_plant_state(state: GardenState, weather: WeatherState, scenario: dict, action_cost: float) -> GardenState:
    ideal = scenario["ideal"]
    moisture_gap = abs(state.soil_moisture - ideal["soil_moisture"])
    nutrient_gap = abs(state.nutrients - ideal["nutrients"])
    temp_gap = abs(state.temperature - ideal["temperature"]) / 15.0
    humidity_gap = abs(state.humidity - ideal["humidity"])
    light_gap = abs(state.light - ideal["light"])

    energy_gap = max(0.0, 0.35 - state.energy_reserve)
    stress = (moisture_gap + nutrient_gap + temp_gap + humidity_gap + light_gap + energy_gap) / 6.0
    growth_gain = max(0.0, 0.09 - stress * 0.12)
    health_delta = 0.03 - stress * 0.09 - action_cost

    state.growth = clamp(state.growth + growth_gain, 0.0, 1.5)
    state.health = clamp(state.health + health_delta, 0.0, 1.0)
    return state
