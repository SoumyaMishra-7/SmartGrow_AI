from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.spec_env import SpecPlantEnv
from strict_policy_agent import get_action
from tasks.tasks import task_easy, task_medium, task_hard

_ = (SpecPlantEnv, get_action)


def collect_scores() -> dict[str, float]:
    easy = task_easy()
    medium = task_medium()
    hard = task_hard()
    avg = round((easy + medium + hard) / 3.0, 4)
    return {
        "easy": round(easy, 4),
        "medium": round(medium, 4),
        "hard": round(hard, 4),
        "average": avg,
    }


def run_all() -> dict[str, float]:
    print("Running baseline agent...\n")

    scores = collect_scores()

    print(f"Easy Score   : {scores['easy']:.2f}")
    print(f"Medium Score : {scores['medium']:.2f}")
    print(f"Hard Score   : {scores['hard']:.2f}")
    print(f"Average      : {scores['average']:.2f}")
    print("\nJSON Output:")
    print(json.dumps(scores, sort_keys=True))

    return scores


if __name__ == "__main__":
    run_all()
