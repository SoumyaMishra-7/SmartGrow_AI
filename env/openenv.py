from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from env.models import ManagementAction, OpenEnvActionModel, OpenEnvResetModel, OpenEnvStateModel, OpenEnvStepModel
from env.resource import ACTIONS
from env.smart_env import SmartGrowEnv

OPEN_ENV_SPEC_PATH = Path(__file__).resolve().parents[1] / "config" / "openenv.yaml"


def _load_openenv_spec() -> dict:
    if not OPEN_ENV_SPEC_PATH.exists():
        raise FileNotFoundError(f"Missing OpenEnv spec file: {OPEN_ENV_SPEC_PATH}")
    spec = yaml.safe_load(OPEN_ENV_SPEC_PATH.read_text(encoding="utf-8"))
    spec["action_count"] = len(ACTIONS)
    return spec


def _build_state_model(env: SmartGrowEnv) -> OpenEnvStateModel:
    state = env.state_snapshot()
    return OpenEnvStateModel(
        day=state.day,
        soil_moisture=state.soil_moisture,
        nutrients=state.nutrients,
        temperature=state.temperature,
        humidity=state.humidity,
        light=state.light,
        growth=state.growth,
        health=state.health,
        water_tank=state.water_tank,
        nutrient_tank=state.nutrient_tank,
        energy_reserve=state.energy_reserve,
        last_action=env.last_action,
    )


class OpenEnvAdapter:
    def __init__(self, scenario_name: str = "balanced", max_days: int = 30, seed: int = 7) -> None:
        self._env = SmartGrowEnv(scenario_name=scenario_name, max_days=max_days, seed=seed)

    @property
    def actions(self) -> list[OpenEnvActionModel]:
        return [
            OpenEnvActionModel(id=action_id, name=effect.label, description=effect.label.replace("_", " "))
            for action_id, effect in sorted(ACTIONS.items())
        ]

    def reset(self, seed: int | None = None) -> OpenEnvResetModel:
        observation, info = self._env.reset(seed=seed)
        return OpenEnvResetModel(observation=observation, info=info, state=self.state())

    def step(self, action: int | dict[str, Any] | ManagementAction) -> OpenEnvStepModel:
        result = self._env.step(action)
        return OpenEnvStepModel(
            observation=result.observation,
            reward=result.reward,
            terminated=result.terminated,
            truncated=result.truncated,
            info=result.info,
            state=self.state(),
        )

    def state(self) -> OpenEnvStateModel:
        return _build_state_model(self._env)

    def spec(self) -> dict:
        return _load_openenv_spec()

    def render(self) -> str:
        return self._env.render()


def make_env(scenario_name: str = "balanced", max_days: int = 30, seed: int = 7) -> OpenEnvAdapter:
    return OpenEnvAdapter(scenario_name=scenario_name, max_days=max_days, seed=seed)


def state_as_dict(env: OpenEnvAdapter) -> dict:
    return asdict(env.state())
