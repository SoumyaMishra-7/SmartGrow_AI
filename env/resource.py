from __future__ import annotations

from env.models import ActionEffect


ACTIONS = {
    0: ActionEffect(label="observe"),
    1: ActionEffect(label="water_light", water_delta=0.10),
    2: ActionEffect(label="water_and_feed", water_delta=0.12, nutrient_delta=0.10),
    3: ActionEffect(label="deep_water", water_delta=0.24),
    4: ActionEffect(label="feed_boost", nutrient_delta=0.18),
    5: ActionEffect(label="shade_canopy", shade_delta=0.20, energy_delta=0.08),
    6: ActionEffect(label="ventilate", fan_delta=0.22, energy_delta=0.10),
    7: ActionEffect(label="climate_control", water_delta=0.08, shade_delta=0.14, fan_delta=0.16, energy_delta=0.18),
}


def action_name(action_id: int) -> str:
    return ACTIONS[action_id].label


def action_effect(action_id: int) -> ActionEffect:
    return ACTIONS[action_id]
