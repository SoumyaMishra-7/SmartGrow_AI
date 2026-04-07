from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Iterable

from agent.strict_policy_agent import get_action
from env.grader import grade_easy, grade_hard, grade_medium
from env.spec_env import Action, SpecPlantEnv
from models import InternalState
from tasks.scenarios import DEFAULT_SCENARIO, SCENARIOS
from utils.helpers import clamp


@dataclass(frozen=True, slots=True)
class TaskSpec:
    name: str
    difficulty: str
    description: str
    scenario_name: str
    max_days: int
    seed: int
    success_hint: str


@dataclass(frozen=True, slots=True)
class EpisodeOutcome:
    total_reward: float
    final_growth: float
    final_health: float
    days_survived: int
    average_growth: float
    average_health: float
    average_water: float
    average_nutrients: float
    completed: bool
    failed: bool


TASKS = {
    "easy_balanced_growth": TaskSpec(
        name="easy_balanced_growth",
        difficulty="easy",
        description="Grow a healthy plant in balanced weather and reach harvest condition.",
        scenario_name="balanced",
        max_days=24,
        seed=11,
        success_hint="Keep growth climbing without letting health drop.",
    ),
    "medium_heat_resilience": TaskSpec(
        name="medium_heat_resilience",
        difficulty="medium",
        description="Protect the plant through a hot, dry period while preserving enough health to finish strongly.",
        scenario_name="hot_dry",
        max_days=30,
        seed=17,
        success_hint="Water and climate control matter more when heat stress rises.",
    ),
    "hard_storm_recovery": TaskSpec(
        name="hard_storm_recovery",
        difficulty="hard",
        description="Recover from unstable stormy weather and still deliver a healthy late-stage plant.",
        scenario_name="stormy",
        max_days=36,
        seed=23,
        success_hint="Avoid over-correcting. Stability matters as much as raw growth.",
    ),
}


def _score_balanced_growth(outcome: EpisodeOutcome) -> float:
    score = (
        outcome.final_growth * 0.45
        + outcome.final_health * 0.30
        + outcome.average_health * 0.15
        + (1.0 if outcome.completed else 0.0) * 0.10
    )
    if outcome.failed:
        score *= 0.35
    return clamp(score, 0.0, 1.0)


def _score_heat_resilience(outcome: EpisodeOutcome) -> float:
    resource_stability = (outcome.average_water + outcome.average_nutrients) / 2.0
    survival_ratio = clamp(outcome.days_survived / TASKS["medium_heat_resilience"].max_days, 0.0, 1.0)
    score = (
        outcome.final_growth * 0.30
        + outcome.final_health * 0.25
        + resource_stability * 0.20
        + survival_ratio * 0.15
        + (1.0 if outcome.completed else 0.0) * 0.10
    )
    if outcome.failed:
        score *= 0.30
    return clamp(score, 0.0, 1.0)


def _score_storm_recovery(outcome: EpisodeOutcome) -> float:
    stability = (outcome.average_health + outcome.average_water + outcome.average_nutrients) / 3.0
    score = (
        outcome.final_growth * 0.25
        + outcome.final_health * 0.25
        + outcome.average_growth * 0.10
        + stability * 0.25
        + (1.0 if outcome.completed else 0.0) * 0.15
    )
    if outcome.failed:
        score *= 0.25
    return clamp(score, 0.0, 1.0)


TASK_GRADERS = {
    "easy_balanced_growth": _score_balanced_growth,
    "medium_heat_resilience": _score_heat_resilience,
    "hard_storm_recovery": _score_storm_recovery,
}


def list_tasks() -> list[str]:
    return sorted(TASKS)


def get_task(name: str) -> TaskSpec:
    return TASKS.get(name, TASKS["easy_balanced_growth"])


def get_task_config(name: str) -> dict:
    scenario_name = name if name in SCENARIOS else get_task(name).scenario_name if name in TASKS else DEFAULT_SCENARIO
    return deepcopy(SCENARIOS[scenario_name])


def grade_task(task_name: str, outcome: EpisodeOutcome) -> float:
    grader = TASK_GRADERS[task_name]
    return round(grader(outcome), 4)


def _run_deterministic_episode(initial_state: dict[str, int], max_steps: int) -> tuple[list[dict[str, int]], int]:
    env = SpecPlantEnv(initial_state=initial_state, max_steps=max_steps)
    obs = env.reset()
    trajectory: list[dict[str, int]] = [obs.dict()]
    action_count = 0

    while True:
        action_dict = get_action(obs.dict())
        action = Action(action=action_dict["action"])
        if action.action != 0:
            action_count += 1
        obs, _, done, _ = env.step(action)
        trajectory.append(obs.dict())
        if done:
            break

    return trajectory, action_count


def _average(values: Iterable[int]) -> float:
    values_list = list(values)
    return float(sum(values_list) / len(values_list)) if values_list else 0.0


def _clamp_score(score: float) -> float:
    return round(clamp(score, 0.0, 1.0), 4)


def task_easy() -> float:
    initial = {"water_level": 58, "sunlight": 56, "plant_health": 84, "growth_stage": 1}
    max_steps = 16
    trajectory, _ = _run_deterministic_episode(initial_state=initial, max_steps=max_steps)

    steps = max(1, len(trajectory) - 1)
    total_health = sum(step["plant_health"] for step in trajectory[1:])
    score = total_health / (steps * 100.0)
    return _clamp_score(score)


def task_medium() -> float:
    initial = {"water_level": 44, "sunlight": 48, "plant_health": 74, "growth_stage": 1}
    max_steps = 20
    trajectory, _ = _run_deterministic_episode(initial_state=initial, max_steps=max_steps)

    steps = max(1, len(trajectory) - 1)
    efficiency_score = sum(
        1
        for step in trajectory[1:]
        if 40 <= step["water_level"] <= 70 and 40 <= step["sunlight"] <= 70
    )
    score = efficiency_score / steps
    return _clamp_score(score)


def task_hard() -> float:
    initial = {"water_level": 38, "sunlight": 42, "plant_health": 66, "growth_stage": 0}
    max_steps = 24
    trajectory, _ = _run_deterministic_episode(initial_state=initial, max_steps=max_steps)

    final_growth_stage = trajectory[-1]["growth_stage"]
    score = final_growth_stage / 5.0
    return _clamp_score(score)
