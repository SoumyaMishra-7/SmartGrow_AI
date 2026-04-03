from dataclasses import replace

from env.reward import calculate_reward
from env.state import GardenState


def test_reward_improves_when_growth_and_health_improve() -> None:
    previous = GardenState(growth=0.3, health=0.6)
    current = replace(previous, growth=0.42, health=0.7)
    assert calculate_reward(previous, current, False) > 0


def test_reward_penalizes_failure() -> None:
    previous = GardenState(growth=0.5, health=0.3)
    current = replace(previous, health=0.1)
    assert calculate_reward(previous, current, True) < 0
