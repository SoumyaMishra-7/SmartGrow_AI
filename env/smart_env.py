from __future__ import annotations

import random
from dataclasses import replace
from typing import Any

from env.models import ManagementAction, StepResult
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

    def reset(self, seed: int | None = None) -> tuple[dict[str, float | int | str], dict]:
        if seed is not None:
            self.rng.seed(seed)
        self.state = GardenState()
        self.last_action = "observe"
        return self.state.observation_dict(), {"scenario": self.scenario_name, "mode": "decision_support"}

    def state_snapshot(self) -> GardenState:
        return replace(self.state)

    def _coerce_action(self, action: int | dict[str, Any] | ManagementAction) -> ManagementAction:
        if isinstance(action, ManagementAction):
            return action

        if isinstance(action, int):
            effect = action_effect(action)
            sunlight_adjustment = "maintain"
            if effect.shade_delta > 0.15:
                sunlight_adjustment = "decrease"
            elif effect.fan_delta > 0.18:
                sunlight_adjustment = "increase"
            nutrients = "add" if effect.nutrient_delta > 0.01 else "hold"
            energy_mode = "high" if effect.energy_delta > 0.14 else "normal"
            return ManagementAction(
                water_ml=int(effect.water_delta * 1000),
                sunlight_adjustment=sunlight_adjustment,
                nutrients=nutrients,
                energy_mode=energy_mode,
            )

        water_ml = int(action.get("water_ml", 0))
        sunlight_adjustment = str(action.get("sunlight_adjustment", "maintain")).lower()
        nutrients = str(action.get("nutrients", "hold")).lower()
        energy_mode = str(action.get("energy_mode", "normal")).lower()
        return ManagementAction(
            water_ml=max(0, min(400, water_ml)),
            sunlight_adjustment=sunlight_adjustment if sunlight_adjustment in {"increase", "decrease", "maintain"} else "maintain",
            nutrients=nutrients if nutrients in {"add", "hold"} else "hold",
            energy_mode=energy_mode if energy_mode in {"low", "normal", "high"} else "normal",
        )

    def step(self, action: int | dict[str, Any] | ManagementAction) -> StepResult:
        previous = replace(self.state)
        weather = generate_weather(self.state.day, self.scenario, self.rng)
        decision = self._coerce_action(action)

        sunlight_factor = {"increase": 0.08, "decrease": -0.10, "maintain": 0.0}[decision.sunlight_adjustment]
        nutrient_delta = 0.08 if decision.nutrients == "add" else 0.0
        energy_scale = {"low": 0.75, "normal": 1.0, "high": 1.2}[decision.energy_mode]

        water_delta = decision.water_ml / 1000.0
        fan_delta = 0.10 if decision.energy_mode in {"normal", "high"} else 0.04
        shade_delta = 0.10 if decision.sunlight_adjustment == "decrease" else 0.0
        energy_delta = energy_scale * (0.04 + fan_delta * 0.22 + shade_delta * 0.18)

        self.state.day += 1
        self.state.temperature = weather.temperature - (shade_delta * 3.0) - (fan_delta * 2.4)
        self.state.humidity = clamp(weather.humidity + water_delta * 0.22, 0.0, 1.0)
        self.state.light = clamp(weather.light + sunlight_factor - shade_delta * 0.2, 0.0, 1.0)
        self.state.soil_moisture = clamp(
            self.state.soil_moisture + weather.rainfall + water_delta - 0.11 - max(0.0, self.state.temperature - 26.0) * 0.012,
            0.0,
            1.0,
        )
        self.state.nutrients = clamp(self.state.nutrients + nutrient_delta - 0.055, 0.0, 1.0)
        self.state.water_tank = clamp(self.state.water_tank - water_delta * 0.7, 0.0, 1.0)
        self.state.nutrient_tank = clamp(self.state.nutrient_tank - nutrient_delta * 0.9, 0.0, 1.0)
        self.state.energy_reserve = clamp(self.state.energy_reserve - energy_delta + weather.light * 0.035, 0.0, 1.0)

        action_cost = (water_delta + nutrient_delta) * 0.07 + energy_delta * 0.11
        self.state = update_plant_state(self.state, weather, self.scenario, action_cost).clipped()
        self.last_action = (
            f"water={decision.water_ml}ml|sun={decision.sunlight_adjustment}|"
            f"nutrients={decision.nutrients}|energy={decision.energy_mode}"
        )

        terminated = self.state.day >= self.max_days or self.state.growth >= 1.0 or self.state.health <= 0.15
        usage = {
            "water_ml": float(decision.water_ml),
            "nutrient_ml": 80.0 if decision.nutrients == "add" else 0.0,
            "energy_wh": round(energy_delta * 1000.0, 2),
        }
        reward = calculate_reward(previous, self.state, terminated, usage=usage)

        expected_outcome = "stabilize"
        if decision.water_ml >= 200:
            expected_outcome = "recover_moisture"
        if decision.nutrients == "add":
            expected_outcome = "improve_growth"
        if decision.energy_mode == "low":
            expected_outcome = "save_energy"

        rationale = (
            f"Selected irrigation={decision.water_ml}ml, sunlight={decision.sunlight_adjustment}, "
            f"nutrients={decision.nutrients}, energy_mode={decision.energy_mode} "
            f"to target healthy growth with constrained resources."
        )

        info = {
            "day": self.state.day,
            "growth": round(self.state.growth, 3),
            "health": round(self.state.health, 3),
            "soil_moisture": round(self.state.soil_moisture, 3),
            "nutrients": round(self.state.nutrients, 3),
            "energy_reserve": round(self.state.energy_reserve, 3),
            "temperature": round(self.state.temperature, 3),
            "action": self.last_action,
            "decision": {
                "water_ml": decision.water_ml,
                "sunlight_adjustment": decision.sunlight_adjustment,
                "nutrients": decision.nutrients,
                "energy_mode": decision.energy_mode,
            },
            "resource_usage": usage,
            "reasoning": rationale,
            "expected_outcome": expected_outcome,
        }
        return StepResult(self.state.observation_dict(), reward, terminated, False, info)

    def render(self) -> str:
        state = self.state
        return (
            f"Day {state.day:02d} | action={self.last_action:<11} | growth={state.growth:.2f} | "
            f"health={state.health:.2f} | moisture={state.soil_moisture:.2f} | nutrients={state.nutrients:.2f} | "
            f"energy={state.energy_reserve:.2f}"
        )
