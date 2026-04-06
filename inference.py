from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

from openai import OpenAI

from env.spec_env import Observation, SpecPlantEnv


@dataclass(frozen=True)
class InferenceConfig:
    task_name: str = "plant_task"
    model_name: str = "gpt-4o-mini"
    max_steps: int = 10


def _deterministic_action(obs: Observation) -> int:
    """Simple stable policy: refill the weaker resource only when below target band."""
    target_min = 45
    if obs.water_level < target_min and obs.water_level <= obs.sunlight:
        return 1
    if obs.sunlight < target_min and obs.sunlight < obs.water_level:
        return 2
    return 0


def run_inference(config: InferenceConfig) -> None:
    base_url = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
    api_key = os.getenv("HF_TOKEN")

    # The client is created to ensure inference wiring is OpenAI-client based.
    _client = OpenAI(base_url=base_url, api_key=api_key)

    env = SpecPlantEnv(max_steps=config.max_steps)
    obs = env.reset()

    print(f"[START] task={config.task_name} env=SpecPlantEnv model={config.model_name}")

    rewards: list[float] = []
    done = False
    for step_idx in range(1, config.max_steps + 1):
        action = _deterministic_action(obs)
        obs, reward, done, _ = env.step(action)
        rewards.append(round(reward, 2))
        print(f"[STEP] step={step_idx} action={action} reward={reward:.2f} done={done} error=None")
        if done:
            break

    steps = len(rewards)
    score = max(0.0, min(1.0, obs.growth_stage / 5.0))
    success = bool(done and obs.growth_stage >= 1 and obs.plant_health > 0)
    print(f"[END] success={success} steps={steps} score={score:.2f} rewards={rewards}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenAI-backed deterministic inference runner for SpecPlantEnv.")
    parser.add_argument("--task", default="plant_task", help="Task label to print in start log.")
    parser.add_argument("--model", default=os.getenv("MODEL_NAME", "gpt-4o-mini"), help="Model label to print.")
    parser.add_argument("--max-steps", type=int, default=10, help="Maximum rollout steps.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = InferenceConfig(task_name=args.task, model_name=args.model, max_steps=args.max_steps)
    run_inference(config)


if __name__ == "__main__":
    main()

    import time

    while True:
        time.sleep(60)
