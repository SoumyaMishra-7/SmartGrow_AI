from __future__ import annotations

from training.evaluate import EvaluationResult
from training.train_dqn import TrainingOutput
from ui.components import section, stat_line
from ui.visuals import mini_bar


def print_training_report(training_output: TrainingOutput, evaluation: EvaluationResult) -> None:
    summary = training_output.metrics.summary()
    training_block = section(
        "Training Summary",
        [
            stat_line("Scenario", training_output.config.scenario_name),
            stat_line("Episodes", training_output.config.episodes),
            stat_line("Avg reward", summary["reward"]),
            stat_line("Avg loss", summary["loss"]),
            stat_line("Avg growth", f"{summary['growth']} {mini_bar(summary['growth'])}"),
            stat_line("Avg health", f"{summary['health']} {mini_bar(summary['health'])}"),
            stat_line("Reward curve", training_output.reward_plot_path or "not saved"),
        ],
    )
    eval_block = section(
        "Evaluation",
        [
            stat_line("Days survived", evaluation.days_survived),
            stat_line("Reward", evaluation.total_reward),
            stat_line("Final growth", f"{evaluation.final_growth} {mini_bar(evaluation.final_growth)}"),
            stat_line("Final health", f"{evaluation.final_health} {mini_bar(evaluation.final_health)}"),
            stat_line("Last state", evaluation.trace[-1]),
        ],
    )
    print(training_block)
    print()
    print(eval_block)
