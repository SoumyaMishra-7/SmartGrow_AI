from agent.strict_policy_agent import decide_action


def test_returns_required_json_shape() -> None:
    result = decide_action({"water_level": 60, "sunlight": 60, "plant_health": 85, "growth_stage": 2})
    assert set(result) == {"action", "reason"}
    assert result["action"] in {0, 1, 2}
    assert isinstance(result["reason"], str)
    assert result["reason"]


def test_critical_low_water_prioritized() -> None:
    result = decide_action({"water_level": 30, "sunlight": 65, "plant_health": 80, "growth_stage": 1})
    assert result["action"] == 1


def test_critical_low_sunlight_prioritized() -> None:
    result = decide_action({"water_level": 65, "sunlight": 30, "plant_health": 80, "growth_stage": 1})
    assert result["action"] == 2


def test_preventive_water_top_up() -> None:
    result = decide_action({"water_level": 42, "sunlight": 58, "plant_health": 80, "growth_stage": 2})
    assert result["action"] == 1


def test_preventive_sunlight_top_up() -> None:
    result = decide_action({"water_level": 58, "sunlight": 42, "plant_health": 80, "growth_stage": 2})
    assert result["action"] == 2


def test_future_aware_water_decay_prevention() -> None:
    result = decide_action({"water_level": 44, "sunlight": 60, "plant_health": 80, "growth_stage": 2})
    assert result["action"] == 1
    assert "next step" in result["reason"].lower()


def test_future_aware_sunlight_decay_prevention() -> None:
    result = decide_action({"water_level": 60, "sunlight": 44, "plant_health": 80, "growth_stage": 2})
    assert result["action"] == 2
    assert "next step" in result["reason"].lower()


def test_stable_range_no_action() -> None:
    result = decide_action({"water_level": 60, "sunlight": 65, "plant_health": 90, "growth_stage": 3})
    assert result["action"] == 0


def test_avoid_overwatering_when_water_high() -> None:
    result = decide_action({"water_level": 80, "sunlight": 55, "plant_health": 88, "growth_stage": 3})
    assert result["action"] != 1


def test_avoid_oversunlight_when_light_high() -> None:
    result = decide_action({"water_level": 55, "sunlight": 80, "plant_health": 88, "growth_stage": 3})
    assert result["action"] != 2
