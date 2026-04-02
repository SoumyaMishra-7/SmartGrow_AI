from __future__ import annotations

import random
from dataclasses import replace

from env.models import StepResult
from env.plant import update_plant_state
from env.resource import ACTIONS, action_effect, action_name
from env.reward import calculate_reward
from env.state import GardenState
from env.weather import generate_weather
from tasks.tasks import get_task_config
from utils.helpers import clamp


class SmartGrowEnv:
    def __init__(self, scenario_name: str = "balanced", max_days: int = 30, seed: int = 7) -> None:
        self.scenario_name = scenario_name
        self.scenario = get_task_config(scenario_name)
        self.max_days = max_days
        self.rng = random.Random(seed)
        self.state = GardenState()
        self.last_action = "observe"

    @property
    def action_space(self) -> tuple[int, ...]:
        return tuple(sorted(ACTIONS))

    def reset(self, seed: int | None = None) -> tuple[tuple[int, ...], dict]:
        if seed is not None:
            self.rng.seed(seed)
        self.state = GardenState()
        self.last_action = "observe"
        return self.state.observation(), {"scenario": self.scenario_name}

    def state_snapshot(self) -> GardenState:
        return replace(self.state)

    def step(self, action_id: int) -> StepResult:
        previous = replace(self.state)
        weather = generate_weather(self.state.day, self.scenario, self.rng)
        effect = action_effect(action_id)

        self.state.day += 1
        self.state.temperature = weather.temperature - (effect.shade_delta * 4.0) - (effect.fan_delta * 3.0)
        self.state.humidity = clamp(weather.humidity + effect.water_delta * 0.25, 0.0, 1.0)
        self.state.light = clamp(weather.light - effect.shade_delta * 0.35, 0.0, 1.0)
        self.state.soil_moisture = clamp(
            self.state.soil_moisture + weather.rainfall + effect.water_delta - 0.12 - max(0.0, self.state.temperature - 26.0) * 0.01,
            0.0,
            1.0,
        )
        self.state.nutrients = clamp(self.state.nutrients + effect.nutrient_delta - 0.06, 0.0, 1.0)
        self.state.water_tank = clamp(self.state.water_tank - effect.water_delta * 0.55, 0.0, 1.0)
        self.state.nutrient_tank = clamp(self.state.nutrient_tank - effect.nutrient_delta * 0.8, 0.0, 1.0)
        self.state.energy_reserve = clamp(self.state.energy_reserve - effect.energy_delta + weather.light * 0.04, 0.0, 1.0)

        action_cost = (effect.water_delta + effect.nutrient_delta) * 0.08 + effect.energy_delta * 0.12
        self.state = update_plant_state(self.state, weather, self.scenario, action_cost).clipped()
        self.last_action = action_name(action_id)

        terminated = self.state.day >= self.max_days or self.state.growth >= 1.0 or self.state.health <= 0.15
        reward = calculate_reward(previous, self.state, terminated)
        info = {
            "day": self.state.day,
            "growth": round(self.state.growth, 3),
            "health": round(self.state.health, 3),
            "soil_moisture": round(self.state.soil_moisture, 3),
            "nutrients": round(self.state.nutrients, 3),
            "energy_reserve": round(self.state.energy_reserve, 3),
            "temperature": round(self.state.temperature, 3),
            "action": self.last_action,
        }
        return StepResult(self.state.observation(), reward, terminated, False, info)

    def render(self) -> str:
        state = self.state
        return (
            f"Day {state.day:02d} | action={self.last_action:<11} | growth={state.growth:.2f} | "
            f"health={state.health:.2f} | moisture={state.soil_moisture:.2f} | nutrients={state.nutrients:.2f} | "
            f"energy={state.energy_reserve:.2f}"
        )
