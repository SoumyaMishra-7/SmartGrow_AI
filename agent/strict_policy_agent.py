from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PlantState:
    water_level: int
    sunlight: int
    plant_health: int
    growth_stage: int


class StrictPolicyAgent:
    """
    Rule-based policy that follows the exact action constraints and priorities:
    0 = do nothing, 1 = add water, 2 = increase sunlight.
    """

    def decide(self, state: PlantState) -> dict[str, int | str]:
        water = state.water_level
        sunlight = state.sunlight

        # 1) Critical safety checks.
        if water < 40:
            return {"action": 1, "reason": "Water is below optimal threshold"}
        if sunlight < 40:
            return {"action": 2, "reason": "Sunlight is below optimal threshold"}

        # 2) Future-aware checks: prevent next-step decay from crossing below optimal.
        if water - 5 < 40 and sunlight - 5 < 40:
            if water <= sunlight:
                return {"action": 1, "reason": "Water will drop below optimal next step"}
            return {"action": 2, "reason": "Sunlight will drop below optimal next step"}
        if water - 5 < 40:
            return {"action": 1, "reason": "Water will drop below optimal next step"}
        if sunlight - 5 < 40:
            return {"action": 2, "reason": "Sunlight will drop below optimal next step"}

        # 4) Stable state takes precedence when both are comfortably optimal.
        if 45 <= water <= 65 and 45 <= sunlight <= 65:
            return {"action": 0, "reason": "Both water and sunlight are in optimal range"}

        # 3) Avoid overcorrection if a resource is already above the safe upper bound.
        if water > 70 and sunlight <= 70:
            return {"action": 2, "reason": "Water is high, balancing with sunlight support"}
        if sunlight > 70 and water <= 70:
            return {"action": 1, "reason": "Sunlight is high, balancing with water support"}

        # 5) Default to efficient stability when no immediate correction is needed.
        return {"action": 0, "reason": "No immediate correction needed; conserving resources"}


def decide_action(state: dict[str, int]) -> dict[str, int | str]:
    parsed = PlantState(
        water_level=int(state["water_level"]),
        sunlight=int(state["sunlight"]),
        plant_health=int(state["plant_health"]),
        growth_stage=int(state["growth_stage"]),
    )
    return StrictPolicyAgent().decide(parsed)


def get_action(state: dict[str, int]) -> dict[str, int | str]:
    return decide_action(state)
