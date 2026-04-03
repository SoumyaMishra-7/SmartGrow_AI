from __future__ import annotations

from env.state import GardenState


class BaselineAgent:
    """
    Deterministic rule-based baseline used for hackathon scoring.
    """

    def act(self, state: GardenState) -> int:
        if state.soil_moisture < 0.38 and state.nutrients < 0.40:
            return 2
        if state.soil_moisture < 0.34:
            return 3
        if state.nutrients < 0.35:
            return 4
        if state.temperature > 29.0 and state.energy_reserve > 0.18:
            return 7
        if state.temperature > 27.5 and state.energy_reserve > 0.10:
            return 6
        if state.light > 0.82 and state.energy_reserve > 0.10:
            return 5
        if state.health < 0.45 and state.soil_moisture < 0.50:
            return 1
        return 0
