from __future__ import annotations

from dataclasses import dataclass

from utils.helpers import clamp


@dataclass(slots=True)
class GardenState:
    day: int = 0
    soil_moisture: float = 0.55
    nutrients: float = 0.55
    temperature: float = 24.0
    humidity: float = 0.60
    light: float = 0.65
    growth: float = 0.10
    health: float = 0.95
    water_tank: float = 1.0
    nutrient_tank: float = 1.0
    energy_reserve: float = 1.0

    def clipped(self) -> "GardenState":
        self.soil_moisture = clamp(self.soil_moisture, 0.0, 1.0)
        self.nutrients = clamp(self.nutrients, 0.0, 1.0)
        self.humidity = clamp(self.humidity, 0.0, 1.0)
        self.light = clamp(self.light, 0.0, 1.0)
        self.growth = clamp(self.growth, 0.0, 1.5)
        self.health = clamp(self.health, 0.0, 1.0)
        self.water_tank = clamp(self.water_tank, 0.0, 1.0)
        self.nutrient_tank = clamp(self.nutrient_tank, 0.0, 1.0)
        self.energy_reserve = clamp(self.energy_reserve, 0.0, 1.0)
        self.temperature = clamp(self.temperature, 0.0, 45.0)
        return self

    def observation(self) -> tuple[int, ...]:
        return (
            bucket(self.soil_moisture),
            bucket(self.nutrients),
            bucket((self.temperature - 10.0) / 25.0),
            bucket(self.humidity),
            bucket(self.light),
            bucket(self.growth / 1.2),
            bucket(self.health),
            bucket(self.energy_reserve),
        )


def bucket(value: float, bins: int = 5) -> int:
    scaled = clamp(value, 0.0, 0.999999)
    return int(scaled * bins)
