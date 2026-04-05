from __future__ import annotations

from dataclasses import dataclass


def _clamp(value: int, low: int = 0, high: int = 100) -> int:
    return max(low, min(high, value))


@dataclass(frozen=True, slots=True)
class Observation:
    water_level: int
    sunlight: int
    plant_health: int
    growth_stage: int

    def dict(self) -> dict[str, int]:
        return {
            "water_level": self.water_level,
            "sunlight": self.sunlight,
            "plant_health": self.plant_health,
            "growth_stage": self.growth_stage,
        }


@dataclass(frozen=True, slots=True)
class Action:
    action: int


class SpecPlantEnv:
    """
    Deterministic reference environment for reproducible baseline scoring.
    Contract:
    - Observation fields are integers in fixed ranges.
    - Actions: 0 do nothing, 1 add water, 2 increase sunlight.
    - At every step both resources decay by 5.
    """

    def __init__(self, initial_state: dict[str, int] | None = None, max_steps: int = 20) -> None:
        self.max_steps = max_steps
        self._initial_state = initial_state or {
            "water_level": 55,
            "sunlight": 55,
            "plant_health": 80,
            "growth_stage": 0,
        }
        self._state: dict[str, int] = {}
        self._steps = 0
        self._optimal_streak = 0
        self._last_growth_step = -1
        self.reset()

    def reset(self) -> Observation:
        self._state = {
            "water_level": _clamp(int(self._initial_state["water_level"])),
            "sunlight": _clamp(int(self._initial_state["sunlight"])),
            "plant_health": _clamp(int(self._initial_state["plant_health"])),
            "growth_stage": max(0, min(5, int(self._initial_state["growth_stage"]))),
        }
        self._steps = 0
        self._optimal_streak = 0
        self._last_growth_step = -1
        return self._observation()

    def _observation(self) -> Observation:
        return Observation(
            water_level=self._state["water_level"],
            sunlight=self._state["sunlight"],
            plant_health=self._state["plant_health"],
            growth_stage=self._state["growth_stage"],
        )

    @staticmethod
    def _in_optimal(value: int) -> bool:
        return 40 <= value <= 70

    def step(self, action: int | Action) -> tuple[Observation, float, bool, dict[str, int | bool | str]]:
        action_id = action.action if isinstance(action, Action) else int(action)
        if action_id not in (0, 1, 2):
            raise ValueError("Action must be 0, 1, or 2")

        before = self._observation()

        if action_id == 1:
            self._state["water_level"] = _clamp(self._state["water_level"] + 10)
        elif action_id == 2:
            self._state["sunlight"] = _clamp(self._state["sunlight"] + 10)

        self._state["water_level"] = _clamp(self._state["water_level"] - 5)
        self._state["sunlight"] = _clamp(self._state["sunlight"] - 5)

        water_ok = self._in_optimal(self._state["water_level"])
        sun_ok = self._in_optimal(self._state["sunlight"])
        both_ok = water_ok and sun_ok

        health_change = 0
        if both_ok:
            health_change += 3
            self._optimal_streak += 1
        else:
            self._optimal_streak = 0
            if not water_ok:
                gap = 40 - self._state["water_level"] if self._state["water_level"] < 40 else self._state["water_level"] - 70
                health_change -= max(2, gap // 6 + 1)
            if not sun_ok:
                gap = 40 - self._state["sunlight"] if self._state["sunlight"] < 40 else self._state["sunlight"] - 70
                health_change -= max(2, gap // 6 + 1)

        if action_id != 0 and both_ok:
            health_change -= 1

        self._state["plant_health"] = _clamp(self._state["plant_health"] + health_change)

        growth_gain = 0
        if both_ok and self._state["plant_health"] >= 50 and self._state["growth_stage"] < 5:
            if self._optimal_streak >= 2 and self._steps != self._last_growth_step:
                self._state["growth_stage"] += 1
                self._last_growth_step = self._steps
                growth_gain = 1
                self._optimal_streak = 0

        self._steps += 1
        done = self._steps >= self.max_steps or self._state["plant_health"] <= 0 or self._state["growth_stage"] >= 5

        after = self._observation()
        reward = float((after.growth_stage - before.growth_stage) * 20 + (after.plant_health - before.plant_health) - (1 if action_id else 0))
        info = {
            "steps": self._steps,
            "water_optimal": water_ok,
            "sunlight_optimal": sun_ok,
            "both_optimal": both_ok,
            "action_name": {0: "do_nothing", 1: "add_water", 2: "increase_sunlight"}[action_id],
        }
        return after, reward, done, info
