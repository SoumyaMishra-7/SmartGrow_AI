from __future__ import annotations

from dataclasses import dataclass, field

from utils.helpers import moving_average


@dataclass(slots=True)
class TrainingMetrics:
    rewards: list[float] = field(default_factory=list)
    growth: list[float] = field(default_factory=list)
    health: list[float] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)

    def record(self, episode_reward: float, final_growth: float, final_health: float, mean_loss: float) -> None:
        self.rewards.append(round(episode_reward, 3))
        self.growth.append(round(final_growth, 3))
        self.health.append(round(final_health, 3))
        self.losses.append(round(mean_loss, 6))

    def summary(self) -> dict[str, float]:
        if not self.rewards:
            return {"reward": 0.0, "growth": 0.0, "health": 0.0, "loss": 0.0}
        return {
            "reward": round(moving_average(self.rewards, 10)[-1], 3),
            "growth": round(moving_average(self.growth, 10)[-1], 3),
            "health": round(moving_average(self.health, 10)[-1], 3),
            "loss": round(moving_average(self.losses, 10)[-1], 6),
        }
