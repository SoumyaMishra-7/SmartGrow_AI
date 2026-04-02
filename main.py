from __future__ import annotations

import argparse

from config import load_runtime_config
from tasks.scenarios import DEFAULT_SCENARIO
from training.config import TrainingConfig
from training.evaluate import evaluate_agent
from training.train_dqn import run_training
from ui.app import print_training_report


def build_parser() -> argparse.ArgumentParser:
    runtime_config = load_runtime_config()
    parser = argparse.ArgumentParser(description="Train SmartGrow AI in a garden simulator.")
    parser.add_argument("--scenario", default=runtime_config.get("scenario", DEFAULT_SCENARIO), help="Scenario name from tasks/scenarios.py")
    parser.add_argument("--episodes", type=int, default=160, help="Number of training episodes")
    parser.add_argument("--days", type=int, default=runtime_config.get("max_days", 30), help="Max days per episode")
    parser.add_argument("--seed", type=int, default=runtime_config.get("seed", 7), help="Random seed")
    parser.add_argument("--learning-rate", type=float, default=runtime_config.get("learning_rate", 0.18), help="Q-learning step size")
    parser.add_argument("--discount", type=float, default=runtime_config.get("discount", 0.94), help="Future reward discount factor")
    parser.add_argument("--epsilon", type=float, default=runtime_config.get("epsilon", 1.0), help="Initial exploration rate")
    parser.add_argument("--epsilon-decay", type=float, default=runtime_config.get("epsilon_decay", 0.985), help="Exploration decay after each episode")
    parser.add_argument("--epsilon-min", type=float, default=runtime_config.get("epsilon_min", 0.05), help="Minimum exploration rate")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = TrainingConfig(
        scenario_name=args.scenario,
        episodes=args.episodes,
        max_days=args.days,
        learning_rate=args.learning_rate,
        discount=args.discount,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
        seed=args.seed,
    )
    training_output = run_training(config)
    evaluation = evaluate_agent(training_output.agent, scenario_name=args.scenario, max_days=args.days, seed=args.seed)
    print_training_report(training_output, evaluation)


if __name__ == "__main__":
    main()
