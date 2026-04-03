from __future__ import annotations

import random
from dataclasses import dataclass

from utils.helpers import clamp


@dataclass(slots=True)
class WeatherState:
    temperature: float
    humidity: float
    light: float
    rainfall: float


def generate_weather(day: int, scenario: dict, rng: random.Random) -> WeatherState:
    phase = (day % 7) / 7.0
    base_temp = scenario["base_temperature"] + 3.0 * (0.5 - abs(phase - 0.5))
    temperature = base_temp + rng.uniform(-scenario["temp_variance"], scenario["temp_variance"])
    humidity = clamp(scenario["base_humidity"] + rng.uniform(-0.15, 0.15), 0.1, 0.95)
    light = clamp(scenario["base_light"] + rng.uniform(-0.18, 0.18), 0.1, 1.0)
    rainfall = clamp(scenario["rain_chance"] if rng.random() < scenario["rain_frequency"] else 0.0, 0.0, 0.35)
    return WeatherState(temperature=temperature, humidity=humidity, light=light, rainfall=rainfall)
