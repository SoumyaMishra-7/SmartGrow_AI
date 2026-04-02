from config.loader import load_runtime_config


def test_runtime_config_loads_expected_defaults() -> None:
    config = load_runtime_config()

    assert config["scenario"] == "balanced"
    assert config["max_days"] == 30
    assert config["seed"] == 7
    assert config["learning_rate"] == 0.18
    assert config["discount"] == 0.94
    assert config["epsilon"] == 1.0
