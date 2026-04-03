from pathlib import Path

from env.openenv import make_env


def test_openenv_adapter_exposes_runtime_spec_from_openenv_yaml() -> None:
    env = make_env()
    spec = env.spec()

    assert Path("config/openenv.yaml").exists()
    assert spec["name"] == "SmartGrowEnv"
    assert spec["action_count"] == 8
    assert "observation_fields" in spec
