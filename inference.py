from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - depends on runtime environment
    OpenAI = None  # type: ignore[assignment]

from env.grader import grade_easy, grade_hard, grade_medium
from env.models import Action, InternalState, Observation
from microfarm_env import MicroFarmEnv


TASKS = ("easy", "medium", "hard")
ENV_NAME = "urban_micro_farm"


@dataclass(frozen=True)
class InferenceConfig:
    model_name: str = os.getenv("MODEL_NAME", "gpt-4o-mini")
    max_steps: int = 8
    api_base_url: str = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
    api_token: str | None = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
    seed: int = 7


def _json_action_string(action: Action) -> str:
    return json.dumps(action.model_dump(), separators=(",", ":"), sort_keys=True)


def _format_bool(value: bool) -> str:
    return "true" if value else "false"


def _format_error(error: str | None) -> str:
    return "null" if error is None else error.replace("\n", " ").strip()


def _compact_rewards(rewards: list[float]) -> str:
    return ",".join(f"{reward:.2f}" for reward in rewards)


def _build_messages(task_name: str, observation: Observation, state: InternalState) -> list[dict[str, str]]:
    reward_payload = state.last_reward.model_dump() if state.last_reward is not None else {"value": 0.0, "reason": "none"}
    return [
        {
            "role": "system",
            "content": "You are an AI assistant managing an urban micro-farm. Optimize plant health while minimizing resource usage.",
        },
        {
            "role": "user",
            "content": (
                f"task={task_name}\n"
                f"step={observation.step}\n"
                f"observation={observation.model_dump_json()}\n"
                f"last_reward={json.dumps(reward_payload, separators=(',', ':'))}\n"
                "Return only valid JSON with keys water_ml, sunlight_adjustment, nutrients, energy_mode."
            ),
        },
    ]


def _fallback_action(observation: Observation, task_name: str) -> Action:
    water_ml = 0.0
    if observation.soil_moisture < 0.40:
        water_ml = 220.0 if task_name == "hard" else 180.0
    elif observation.soil_moisture < 0.55:
        water_ml = 120.0

    if observation.weather == "cloudy":
        sunlight = "increase"
    elif observation.weather == "humid":
        sunlight = "decrease"
    else:
        sunlight = "maintain"

    nutrients = "add" if observation.plant_health < 0.70 and observation.step % 3 == 0 else "none"
    energy_mode = "low" if task_name == "easy" else "normal"
    if task_name == "hard" and observation.energy_budget > 500 and observation.plant_health < 0.75:
        energy_mode = "high"

    return Action(
        water_ml=water_ml,
        sunlight_adjustment=sunlight,
        nutrients=nutrients,
        energy_mode=energy_mode,
    )


def _extract_response_text(response: Any) -> str:
    text = getattr(response, "output_text", "") or ""
    if text:
        return text

    output = getattr(response, "output", None) or []
    for item in output:
        content = getattr(item, "content", None) or []
        for block in content:
            block_text = getattr(block, "text", None)
            if block_text:
                return block_text
    raise ValueError("Model response did not contain text output.")


def _llm_action(client: OpenAI, model_name: str, task_name: str, observation: Observation, state: InternalState) -> Action:
    response = client.responses.create(
        model=model_name,
        input=_build_messages(task_name, observation, state),
        temperature=0,
    )
    content = _extract_response_text(response)
    return Action.model_validate_json(content)


def _select_action(
    client: Any | None,
    model_name: str,
    task_name: str,
    observation: Observation,
    state: InternalState,
) -> tuple[Action, str | None]:
    if client is None:
        return _fallback_action(observation, task_name), "missing_api_token"

    try:
        return _llm_action(client, model_name, task_name, observation, state), None
    except Exception as exc:  # noqa: BLE001
        return _fallback_action(observation, task_name), f"{type(exc).__name__}:{str(exc)[:120]}"


def _score_task(task_name: str, state: InternalState) -> float:
    if task_name == "easy":
        return grade_easy(state)
    if task_name == "medium":
        return grade_medium(state)
    return grade_hard(state)


def _run_task(client: Any | None, config: InferenceConfig, task_name: str) -> None:
    env = MicroFarmEnv(task_type=task_name, seed=config.seed, max_steps=config.max_steps)
    observation = env.reset()
    print(f"[START] task={task_name} env={ENV_NAME} model={config.model_name}")

    rewards: list[float] = []
    success = False
    try:
        for step_idx in range(1, config.max_steps + 1):
            action, error = _select_action(client, config.model_name, task_name, observation, env.state())
            observation, reward, done, _info = env.step(action)
            rewards.append(float(reward))
            print(
                f"[STEP] step={step_idx} action={_json_action_string(action)} "
                f"reward={reward:.2f} done={_format_bool(done)} error={_format_error(error)}"
            )
            if done:
                break

        final_state = env.state()
        score = _score_task(task_name, final_state)
        success = bool(final_state.average_health > 0.20 and final_state.average_growth > 0.15)
        print(
            f"[END] success={_format_bool(success)} steps={len(rewards)} "
            f"score={score:.2f} rewards={_compact_rewards(rewards)}"
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"[END] success={_format_bool(False)} steps={len(rewards)} "
            f"score=0.00 rewards={_compact_rewards(rewards)}"
        )


def run_inference(config: InferenceConfig) -> None:
    client = None
    if OpenAI is not None and config.api_token:
        client = OpenAI(base_url=config.api_base_url, api_key=config.api_token)
    for task_name in TASKS:
        _run_task(client, config, task_name)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Strict OpenEnv inference runner for Urban Micro-Farm AI Assistant.")
    parser.add_argument("--model", default=os.getenv("MODEL_NAME", "gpt-4o-mini"), help="Model label to print and use.")
    parser.add_argument("--max-steps", type=int, default=8, help="Maximum rollout steps per task.")
    parser.add_argument("--seed", type=int, default=7, help="Deterministic seed for environment rollouts.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = InferenceConfig(model_name=args.model, max_steps=args.max_steps, seed=args.seed)
    run_inference(config)


if __name__ == "__main__":
    main()
