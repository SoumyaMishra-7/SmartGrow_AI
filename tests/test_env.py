from env.smart_env import SmartGrowEnv


def test_environment_step_returns_expected_shape() -> None:
    env = SmartGrowEnv(seed=3)
    observation, _ = env.reset()
    assert len(observation) == 8

    step = env.step(1)
    assert isinstance(step.observation, tuple)
    assert isinstance(step.reward, float)
    assert "growth" in step.info
    assert "energy_reserve" in step.info


def test_environment_terminates_within_max_days() -> None:
    env = SmartGrowEnv(max_days=4, seed=1)
    env.reset()
    terminated = False
    for _ in range(8):
        step = env.step(0)
        terminated = step.terminated
        if terminated:
            break
    assert terminated
