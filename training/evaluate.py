from __future__ import annotations

from dataclasses import dataclass

from agent.dqn_agent import DQNAgent
from env.smart_env import SmartGrowEnv


@dataclass(slots=True)
class EvaluationResult:
    total_reward: float
    final_growth: float
    final_health: float
    days_survived: int
    trace: list[str]


def evaluate_agent(agent: DQNAgent, scenario_name: str, max_days: int = 30, seed: int = 99) -> EvaluationResult:
    env = SmartGrowEnv(scenario_name=scenario_name, max_days=max_days, seed=seed)
    observation, _ = env.reset(seed=seed)
    total_reward = 0.0
    trace: list[str] = [env.render()]

    while True:
        action = agent.act(observation, training=False)
        step = env.step(action)
        observation = step.observation
        total_reward += step.reward
        trace.append(env.render())
        if step.terminated or step.truncated:
            break

    return EvaluationResult(
        total_reward=round(total_reward, 3),
        final_growth=round(env.state.growth, 3),
        final_health=round(env.state.health, 3),
        days_survived=env.state.day,
        trace=trace,
    )
