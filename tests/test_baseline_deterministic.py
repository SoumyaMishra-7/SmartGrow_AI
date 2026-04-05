from agent.run_baseline import collect_scores
from strict_policy_agent import get_action
from tasks.tasks import task_easy, task_hard, task_medium


def test_task_scores_are_deterministic() -> None:
    run1 = (task_easy(), task_medium(), task_hard())
    run2 = (task_easy(), task_medium(), task_hard())
    assert run1 == run2


def test_task_scores_are_normalized() -> None:
    scores = (task_easy(), task_medium(), task_hard())
    assert all(0.0 <= score <= 1.0 for score in scores)


def test_collect_scores_is_deterministic_and_structured() -> None:
    run1 = collect_scores()
    run2 = collect_scores()

    assert run1 == run2
    assert set(run1) == {"easy", "medium", "hard", "average"}
    assert all(0.0 <= float(value) <= 1.0 for value in run1.values())


def test_strict_policy_action_is_in_range() -> None:
    result = get_action({"water_level": 50, "sunlight": 50, "plant_health": 80, "growth_stage": 2})
    assert result["action"] in {0, 1, 2}
    assert isinstance(result["reason"], str)
    assert "action" in result
    assert "reason" in result
