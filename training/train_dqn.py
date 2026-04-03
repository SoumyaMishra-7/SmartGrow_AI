from __future__ import annotations

from dataclasses import dataclass

from agent.dqn_agent import DQNAgent
from env.smart_env import SmartGrowEnv
from training.config import TrainingConfig
from utils.metrics import TrainingMetrics
from utils.plotting import save_reward_curve


@dataclass(slots=True)
class TrainingOutput:
    agent: DQNAgent
    metrics: TrainingMetrics
    config: TrainingConfig
    reward_plot_path: str | None = None


def run_training(config: TrainingConfig) -> TrainingOutput:
    env = SmartGrowEnv(scenario_name=config.scenario_name, max_days=config.max_days, seed=config.seed)
    agent = DQNAgent(
        action_size=len(env.action_space),
        learning_rate=config.learning_rate,
        discount=config.discount,
        epsilon=config.epsilon,
        epsilon_decay=config.epsilon_decay,
        epsilon_min=config.epsilon_min,
        seed=config.seed,
    )
    metrics = TrainingMetrics()

    for episode in range(config.episodes):
        observation, _ = env.reset(seed=config.seed + episode)
        episode_reward = 0.0
        episode_losses: list[float] = []

        while True:
            action = agent.act(observation, training=True)
            step = env.step(action)
            loss = agent.learn(observation, action, step.reward, step.observation, step.terminated)
            observation = step.observation
            episode_reward += step.reward
            episode_losses.append(loss)
            if step.terminated or step.truncated:
                mean_loss = sum(episode_losses) / len(episode_losses)
                metrics.record(episode_reward, env.state.growth, env.state.health, mean_loss)
                break

        agent.decay_epsilon()
        if episode == 0 or (episode + 1) % config.log_interval == 0 or episode + 1 == config.episodes:
            print(
                f"episode={episode + 1:03d} reward={episode_reward:7.3f} "
                f"loss={mean_loss:9.6f} epsilon={agent.epsilon:0.3f} action_source=agent.act"
            )

    reward_plot_path = save_reward_curve(metrics.rewards) if config.save_plot else None
    return TrainingOutput(agent=agent, metrics=metrics, config=config, reward_plot_path=reward_plot_path)
