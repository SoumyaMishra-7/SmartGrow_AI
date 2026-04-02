from __future__ import annotations

import argparse
from statistics import mean

from agent.baseline_agent import BaselineAgent
from env.smart_env import SmartGrowEnv
from tasks.tasks import EpisodeOutcome, TASKS, get_task, grade_task, list_tasks


def run_task(task_name: str, seed: int | None = None) -> tuple[EpisodeOutcome, float]:
    task = get_task(task_name)
    run_seed = task.seed if seed is None else seed
    env = SmartGrowEnv(scenario_name=task.scenario_name, max_days=task.max_days, seed=run_seed)
    agent = BaselineAgent()
    env.reset(seed=run_seed)

    growth_values: list[float] = []
    health_values: list[float] = []
    water_values: list[float] = []
    nutrient_values: list[float] = []
    total_reward = 0.0

    while True:
        action_id = agent.act(env.state_snapshot())
        step = env.step(action_id)
        total_reward += step.reward
        growth_values.append(env.state.growth)
        health_values.append(env.state.health)
        water_values.append(env.state.soil_moisture)
        nutrient_values.append(env.state.nutrients)
        if step.terminated or step.truncated:
            break

    outcome = EpisodeOutcome(
        total_reward=round(total_reward, 4),
        final_growth=round(env.state.growth, 4),
        final_health=round(env.state.health, 4),
        days_survived=env.state.day,
        average_growth=round(mean(growth_values), 4),
        average_health=round(mean(health_values), 4),
        average_water=round(mean(water_values), 4),
        average_nutrients=round(mean(nutrient_values), 4),
        completed=env.state.growth >= 1.0 and env.state.health >= 0.65,
        failed=env.state.health <= 0.15,
    )
    return outcome, grade_task(task.name, outcome)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the reproducible baseline inference script across SmartGrow tasks.")
    parser.add_argument("--task", choices=list_tasks(), help="Run a single task instead of the full benchmark.", default=None)
    parser.add_argument("--seed", type=int, default=None, help="Override the default fixed task seed.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    task_names = [args.task] if args.task else list_tasks()
    aggregate_scores: list[float] = []

    for task_name in task_names:
        task = get_task(task_name)
        outcome, score = run_task(task_name, seed=args.seed)
        aggregate_scores.append(score)
        print(
            f"{task.name} difficulty={task.difficulty} scenario={task.scenario_name} "
            f"seed={task.seed if args.seed is None else args.seed} score={score:.4f} "
            f"growth={outcome.final_growth:.4f} health={outcome.final_health:.4f} "
            f"days={outcome.days_survived} reward={outcome.total_reward:.4f}"
        )

    if len(aggregate_scores) > 1:
        print(f"aggregate_score={mean(aggregate_scores):.4f}")


if __name__ == "__main__":
    main()
