from __future__ import annotations

from env.models import ActionEffect


ACTIONS = {
    0: ActionEffect(label="monitor_only"),
    1: ActionEffect(label="precision_irrigation", water_delta=0.10),
    2: ActionEffect(label="irrigate_and_feed", water_delta=0.12, nutrient_delta=0.10),
    3: ActionEffect(label="recovery_irrigation", water_delta=0.24),
    4: ActionEffect(label="nutrient_topup", nutrient_delta=0.18),
    5: ActionEffect(label="reduce_light_load", shade_delta=0.20, energy_delta=0.08),
    6: ActionEffect(label="ventilation_boost", fan_delta=0.22, energy_delta=0.10),
    7: ActionEffect(label="active_microclimate_control", water_delta=0.08, shade_delta=0.14, fan_delta=0.16, energy_delta=0.18),
}


def action_name(action_id: int) -> str:
    return ACTIONS[action_id].label


def action_effect(action_id: int) -> ActionEffect:
    return ACTIONS[action_id]
