from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

from env.models import Action, InternalState, Observation, PlantSnapshot, Reward
from env.plant import update_plant_state
from env.reward import calculate_reward
from env.state import GardenState
from env.weather import generate_weather
from tasks.scenarios import SCENARIOS
from utils.helpers import clamp


@dataclass(frozen=True, slots=True)
class TaskConfig:
    scenario_name: str
    plant_count: int
    max_steps: int
    initial_water_tank: float
    initial_nutrient_tank: float
    initial_energy_reserve: float
    dynamic_weather: bool
    water_decay_scale: float
    nutrient_decay_scale: float
    energy_decay_scale: float


TASK_CONFIGS: dict[str, TaskConfig] = {
    "easy": TaskConfig(
        scenario_name="balanced",
        plant_count=1,
        max_steps=20,
        initial_water_tank=1.0,
        initial_nutrient_tank=1.0,
        initial_energy_reserve=1.0,
        dynamic_weather=False,
        water_decay_scale=1.0,
        nutrient_decay_scale=1.0,
        energy_decay_scale=1.0,
    ),
    "medium": TaskConfig(
        scenario_name="hot_dry",
        plant_count=2,
        max_steps=24,
        initial_water_tank=0.9,
        initial_nutrient_tank=0.9,
        initial_energy_reserve=0.9,
        dynamic_weather=False,
        water_decay_scale=1.18,
        nutrient_decay_scale=1.14,
        energy_decay_scale=1.12,
    ),
    "hard": TaskConfig(
        scenario_name="stormy",
        plant_count=4,
        max_steps=28,
        initial_water_tank=0.72,
        initial_nutrient_tank=0.75,
        initial_energy_reserve=0.78,
        dynamic_weather=True,
        water_decay_scale=1.35,
        nutrient_decay_scale=1.28,
        energy_decay_scale=1.24,
    ),
}


class MicroFarmEnv:
    def __init__(self, task_type: str = "easy", seed: int = 7, max_steps: int | None = None) -> None:
        if task_type not in TASK_CONFIGS:
            raise ValueError(f"Unsupported task_type={task_type!r}. Expected one of {sorted(TASK_CONFIGS)}")

        self.task_type = task_type
        self.seed = seed
        self.rng = random.Random(seed)
        self._task = TASK_CONFIGS[task_type]
        self.max_steps = max_steps or self._task.max_steps
        self._state = GardenState()
        self._weather_label = "sunny"
        self._scenario_name = self._task.scenario_name
        self._terminated = False
        self._last_action: Action | None = None
        self._last_reward = Reward(value=0.0, reason="Environment reset.")
        self.internal_state = InternalState(
            task_type=self.task_type,
            scenario_name=self._scenario_name,
            plant_count=self._task.plant_count,
            seed=self.seed,
            step_count=0,
            max_steps=self.max_steps,
            weather=self._weather_label,
            soil_moisture=0.0,
            nutrients=0.0,
            temperature=0.0,
            humidity=0.0,
            light=0.0,
            water_remaining=0.0,
            nutrient_remaining=0.0,
            energy_budget=0.0,
            average_growth=0.0,
            average_health=0.0,
            plants=[],
            last_action=None,
            last_reward=self._last_reward,
            terminated=False,
        )
        self.reset()

    def reset(self) -> Observation:
        self.rng.seed(self.seed)
        self._terminated = False
        self._last_action = None
        self._last_reward = Reward(value=0.0, reason="Environment reset.")
        self._weather_label = "sunny"
        self._scenario_name = self._task.scenario_name
        self._state = GardenState(
            water_tank=self._task.initial_water_tank,
            nutrient_tank=self._task.initial_nutrient_tank,
            energy_reserve=self._task.initial_energy_reserve,
        ).clipped()
        self.internal_state = self._build_internal_state()
        return self._build_observation()

    def step(self, action: Action | dict[str, Any]) -> tuple[Observation, float, bool, dict[str, Any]]:
        if self._terminated:
            raise RuntimeError("Episode is terminated. Call reset() before step().")

        parsed_action = self._coerce_action(action)
        previous = replace(self._state)
        scenario = self._scenario_for_day(self._state.day)
        weather = generate_weather(self._state.day, scenario, self.rng)
        self._weather_label = self._weather_from_weather_state(weather.light, weather.humidity)
        self._scenario_name = self._active_scenario_name(self._state.day)

        sunlight_factor = {"increase": 0.08, "decrease": -0.10, "maintain": 0.0}[parsed_action.sunlight_adjustment]
        nutrient_delta = 0.08 if parsed_action.nutrients == "add" else 0.0
        energy_scale = {"low": 0.75, "normal": 1.0, "high": 1.2}[parsed_action.energy_mode]
        plant_load = 1.0 + 0.30 * (self._task.plant_count - 1)

        water_delta = parsed_action.water_ml / 1000.0
        fan_delta = 0.10 if parsed_action.energy_mode in {"normal", "high"} else 0.04
        shade_delta = 0.10 if parsed_action.sunlight_adjustment == "decrease" else 0.0
        energy_delta = energy_scale * (0.04 + fan_delta * 0.22 + shade_delta * 0.18) * self._task.energy_decay_scale

        self._state.day += 1
        self._state.temperature = weather.temperature - (shade_delta * 3.0) - (fan_delta * 2.4)
        self._state.humidity = clamp(weather.humidity + (water_delta / plant_load) * 0.22, 0.0, 1.0)
        self._state.light = clamp(weather.light + sunlight_factor - shade_delta * 0.2, 0.0, 1.0)
        self._state.soil_moisture = clamp(
            self._state.soil_moisture
            + weather.rainfall
            + (water_delta / plant_load)
            - 0.11 * self._task.water_decay_scale
            - max(0.0, self._state.temperature - 26.0) * 0.012,
            0.0,
            1.0,
        )
        self._state.nutrients = clamp(
            self._state.nutrients + nutrient_delta - 0.055 * self._task.nutrient_decay_scale,
            0.0,
            1.0,
        )
        self._state.water_tank = clamp(self._state.water_tank - water_delta * (0.7 + 0.08 * (self._task.plant_count - 1)), 0.0, 1.0)
        self._state.nutrient_tank = clamp(
            self._state.nutrient_tank - nutrient_delta * (0.9 + 0.05 * (self._task.plant_count - 1)),
            0.0,
            1.0,
        )
        self._state.energy_reserve = clamp(self._state.energy_reserve - energy_delta + weather.light * 0.035, 0.0, 1.0)

        action_cost = ((water_delta + nutrient_delta) * 0.07 + energy_delta * 0.11) * plant_load
        self._state = update_plant_state(self._state, weather, scenario, action_cost).clipped()

        usage = {
            "water_ml": float(parsed_action.water_ml),
            "nutrient_ml": 80.0 if parsed_action.nutrients == "add" else 0.0,
            "energy_wh": round(energy_delta * 1000.0, 2),
        }

        self._terminated = (
            self._state.day >= self.max_steps
            or self._state.growth >= 1.0
            or self._state.health <= 0.15
            or self._state.water_tank <= 0.01
            or self._state.energy_reserve <= 0.01
        )

        self._last_action = parsed_action
        self._last_reward = self._build_reward(previous, self._state, usage, self._terminated)
        observation = self._build_observation()
        self.internal_state = self._build_internal_state()

        info = {
            "task_type": self.task_type,
            "scenario_name": self._scenario_name,
            "plant_count": self._task.plant_count,
            "reward_reason": self._last_reward.reason,
            "resource_usage": usage,
            "weather": self._weather_label,
            "growth": round(self._state.growth, 3),
            "health": round(self._state.health, 3),
        }
        return observation, self._last_reward.value, self._terminated, info

    def state(self) -> InternalState:
        return self.internal_state

    def _coerce_action(self, action: Action | dict[str, Any]) -> Action:
        if isinstance(action, Action):
            return action

        payload = dict(action)
        if payload.get("nutrients") == "hold":
            payload["nutrients"] = "none"
        return Action.model_validate(payload)

    def _active_scenario_name(self, day: int) -> str:
        if not self._task.dynamic_weather:
            return self._task.scenario_name

        sequence = ("stormy", "hot_dry", "balanced", "hot_dry")
        return sequence[day % len(sequence)]

    def _scenario_for_day(self, day: int) -> dict[str, Any]:
        name = self._active_scenario_name(day)
        scenario = deepcopy(SCENARIOS[name])
        if self.task_type == "easy":
            scenario["temp_variance"] = 0.8
            scenario["rain_frequency"] = 0.18
            scenario["rain_chance"] = 0.08
        elif self.task_type == "medium":
            scenario["temp_variance"] = max(scenario["temp_variance"], 2.8)
            scenario["base_light"] = clamp(scenario["base_light"] - 0.03, 0.1, 1.0)
        return scenario

    def _build_observation(self) -> Observation:
        return Observation(
            plant_health=round(self._state.health, 3),
            soil_moisture=round(self._state.soil_moisture, 3),
            weather=self._weather_label,
            water_remaining=round(self._state.water_tank * 1000.0, 2),
            energy_budget=round(self._state.energy_reserve * 1000.0, 2),
            step=self._state.day,
        )

    def _build_internal_state(self) -> InternalState:
        plants = self._plant_snapshots()
        return InternalState(
            task_type=self.task_type,
            scenario_name=self._scenario_name,
            plant_count=self._task.plant_count,
            seed=self.seed,
            step_count=self._state.day,
            max_steps=self.max_steps,
            weather=self._weather_label,
            soil_moisture=round(self._state.soil_moisture, 4),
            nutrients=round(self._state.nutrients, 4),
            temperature=round(self._state.temperature, 4),
            humidity=round(self._state.humidity, 4),
            light=round(self._state.light, 4),
            water_remaining=round(self._state.water_tank * 1000.0, 2),
            nutrient_remaining=round(self._state.nutrient_tank * 1000.0, 2),
            energy_budget=round(self._state.energy_reserve * 1000.0, 2),
            average_growth=round(self._state.growth, 4),
            average_health=round(self._state.health, 4),
            plants=plants,
            last_action=self._last_action,
            last_reward=self._last_reward,
            terminated=self._terminated,
        )

    def _plant_snapshots(self) -> list[PlantSnapshot]:
        snapshots: list[PlantSnapshot] = []
        spread = 0.015 if self._task.plant_count > 1 else 0.0
        center = (self._task.plant_count - 1) / 2.0
        for plant_index in range(self._task.plant_count):
            offset = (plant_index - center) * spread
            snapshots.append(
                PlantSnapshot(
                    plant_id=plant_index + 1,
                    health=round(clamp(self._state.health - abs(offset), 0.0, 1.0), 4),
                    growth=round(clamp(self._state.growth - max(0.0, abs(offset) * 0.8), 0.0, 1.5), 4),
                )
            )
        return snapshots

    def _build_reward(
        self,
        previous: GardenState,
        current: GardenState,
        usage: dict[str, float],
        done: bool,
    ) -> Reward:
        legacy_reward = calculate_reward(previous, current, done, usage=usage)
        health_gain = max(0.0, current.health - previous.health)
        growth_gain = max(0.0, current.growth - previous.growth)
        damage = max(0.0, previous.health - current.health)
        overuse = (
            max(0.0, usage["water_ml"] - 220.0) / 220.0
            + max(0.0, usage["energy_wh"] - 120.0) / 120.0
            + max(0.0, usage["nutrient_ml"] - 80.0) / 80.0
        ) / 3.0
        reserve_efficiency = (current.water_tank + current.energy_reserve + current.nutrient_tank) / 3.0
        normalized_legacy = clamp((legacy_reward + 2.0) / 16.0, 0.0, 1.0)

        value = clamp(
            normalized_legacy * 0.45
            + clamp((health_gain + growth_gain) * 6.5, 0.0, 1.0) * 0.30
            + reserve_efficiency * 0.20
            - overuse * 0.20
            - clamp(damage * 5.0, 0.0, 1.0) * 0.15,
            0.0,
            1.0,
        )

        reasons: list[str] = []
        if health_gain > 0.0:
            reasons.append("plant health improved")
        if growth_gain > 0.0:
            reasons.append("growth advanced")
        if overuse > 0.0:
            reasons.append("resource overuse penalty applied")
        if damage > 0.0:
            reasons.append("plant damage penalty applied")
        if not reasons:
            reasons.append("state held steady with partial step reward")

        return Reward(value=round(value, 4), reason=", ".join(reasons))

    @staticmethod
    def _weather_from_weather_state(light: float, humidity: float) -> str:
        if light < 0.4:
            return "cloudy"
        if humidity > 0.8:
            return "humid"
        return "sunny"
