# SmartGrow AI

Deterministic, explainable baseline system for a single-plant OpenEnv-style control task.

## Problem Statement

Build an AI policy that maximizes plant growth while preserving health under constrained resource dynamics. The controller must remain stable over time, avoid over-correction, and be fully reproducible for grading.

## Environment Design

The contract-compliant environment is implemented in `env/spec_env.py` as `SpecPlantEnv`.

State:

```json
{
	"water_level": "0-100",
	"sunlight": "0-100",
	"plant_health": "0-100",
	"growth_stage": "0-5"
}
```

Dynamics per step:

- `water_level -= 5`
- `sunlight -= 5`
- Action can increase only one resource at a time
- Health improves when water and sunlight are both in optimal range `[40, 70]`
- Health drops when outside optimal range

Termination:

- `growth_stage >= 5`, or
- `plant_health <= 0`, or
- max step horizon reached

## Action Space

- `0`: Do nothing
- `1`: Add water (`+10 water_level`)
- `2`: Increase sunlight (`+10 sunlight`)

## Observation Space

Each step returns:

```json
{
	"water_level": 0,
	"sunlight": 0,
	"plant_health": 0,
	"growth_stage": 0
}
```

All values are deterministic integer updates.

## Reward Function

Reward is deterministic and combines:

- growth progress reward
- plant health delta reward
- small action-efficiency penalty for non-zero actions

This supports policy stability and reduced unnecessary interventions.

## Tasks (Easy / Medium / Hard)

Deterministic benchmark tasks are implemented in `tasks/tasks.py`:

- `task_easy()`
- `task_medium()`
- `task_hard()`

Each task:

- uses fixed initial conditions
- runs the strict policy baseline
- computes a deterministic normalized score in `[0.0, 1.0]`

Scoring logic:

- Easy: `score = total_health / (steps * 100)`
- Medium: `score = efficiency_score / steps`
- Hard: `score = final_growth_stage / 5`

All task scores are clamped to `[0.0, 1.0]`.

## Baseline Scores

From `python run_baseline.py`:

- Easy Score: `0.9156`
- Medium Score: `1.0000`
- Hard Score: `0.0000`
- Average Score: `0.6385`

These values are reproducible and identical across repeated runs.

Example JSON output:

```json
{
	"easy": 0.9156,
	"medium": 1.0,
	"hard": 0.0,
	"average": 0.6385
}
```

## Determinism Guarantee

- Task rollouts are fully deterministic.
- Policy decisions are deterministic and constrained to actions `{0, 1, 2}`.
- Baseline runs produce identical results for identical inputs.
- Determinism verifier runs the baseline scoring twice and fails if any value differs.

## How to Run

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run deterministic baseline evaluation:

```powershell
python run_baseline.py
```

Run determinism verification:

```powershell
python verify_determinism.py
```

Run tests:

```powershell
python -m pytest -q
```

Use the strict JSON CLI:

```powershell
echo {"water_level":42,"sunlight":43,"plant_health":80,"growth_stage":2} | python cli.py
```

## Real-world Impact

- Demonstrates explainable control for urban micro-farming and vertical gardening.
- Supports low-cost automation where deterministic behavior is required.
- Provides reproducible benchmarking for policy comparisons in hackathon evaluation.
