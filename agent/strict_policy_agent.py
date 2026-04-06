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

    STABLE_MIN = 45
    STABLE_MAX = 65
    CRITICAL_MIN = 40
    DECAY_PER_STEP = 5

    @staticmethod
    def _payload(action: int, reason: str) -> dict[str, int | str]:
        return {"action": action, "reason": reason}

    @staticmethod
    def _projected_after_decay(value: int) -> int:
        return value - StrictPolicyAgent.DECAY_PER_STEP

    @staticmethod
    def _urgency(value: int) -> int:
        if value < StrictPolicyAgent.CRITICAL_MIN:
            return 2
        if value < StrictPolicyAgent.STABLE_MIN:
            return 1
        return 0

    @staticmethod
    def _tie_break(state: PlantState) -> int:
        return 1 if state.growth_stage % 2 == 0 else 2

    def decide(self, state: PlantState) -> dict[str, int | str]:
        water = state.water_level
        sunlight = state.sunlight
        projected_water = self._projected_after_decay(water)
        projected_sunlight = self._projected_after_decay(sunlight)

        # 1) Stability zone: if both resources are already comfortable, conserve action.
        if self.STABLE_MIN <= water <= self.STABLE_MAX and self.STABLE_MIN <= sunlight <= self.STABLE_MAX:
            return self._payload(0, "Both resources are in the stability band")

        # 2) Safety guard: never let either resource enter the critical band.
        water_urgency = self._urgency(water)
        sunlight_urgency = self._urgency(sunlight)

        if water_urgency == 0 and sunlight_urgency == 0:
            if water > 70 and sunlight > 70:
                return self._payload(0, "Both resources are above the upper bound")
            return self._payload(0, "No urgent correction needed")

        if water_urgency > sunlight_urgency:
            reason = "Water is below the critical threshold" if water < self.CRITICAL_MIN else "Water will drop below critical next step"
            return self._payload(1, reason)
        if sunlight_urgency > water_urgency:
            reason = "Sunlight is below the critical threshold" if sunlight < self.CRITICAL_MIN else "Sunlight will drop below critical next step"
            return self._payload(2, reason)

        # 3) Tie-break deterministically when both resources are equally urgent.
        if projected_water != projected_sunlight:
            return self._payload(1 if projected_water < projected_sunlight else 2, "Choose the lower projected resource")
        return self._payload(self._tie_break(state), "Tie broken deterministically to preserve balance")


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
