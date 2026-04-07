---
title: Urban Micro-Farm AI Assistant
sdk: streamlit
app_file: app.py
---

# 🌱 Urban Micro-Farm AI Assistant (OpenEnv)

An OpenEnv-compliant environment for training AI agents to manage urban farming systems under real-world constraints.

Designed for OpenEnv Hackathon Submission.

## Problem Description

This project reframes the original gardening simulation as an AI Assistant for Urban Micro-Farm Resource Optimization under Real-World Constraints.

Each step represents a practical operating decision:
- water allocation
- sunlight control
- nutrient management
- energy optimization

The objective is to maintain plant health and growth while minimizing waste in a constrained urban farming system.

## 🚀 Real-World Impact

This environment simulates real-world urban farming decision-making under resource constraints.

It can be used for:
- smart city farming systems
- IoT-driven agriculture automation
- sustainability optimization for water and energy usage
- AI decision-making research

Unlike toy simulations, this environment models:
- resource scarcity
- environmental variability
- multi-objective optimization

## Observation Space

The environment returns a typed OpenEnv observation:

```json
{
  "plant_health": 0.95,
  "soil_moisture": 0.55,
  "weather": "sunny",
  "water_remaining": 1000.0,
  "energy_budget": 1000.0,
  "step": 0
}
```

Fields:
- `plant_health`: normalized plant health
- `soil_moisture`: normalized moisture level
- `weather`: current weather label
- `water_remaining`: remaining water budget
- `energy_budget`: remaining energy budget
- `step`: current episode step

## Action Space

Each step accepts a structured action:

```json
{
  "water_ml": 200,
  "sunlight_adjustment": "increase",
  "nutrients": "add",
  "energy_mode": "low"
}
```

Fields:
- `water_ml`: irrigation amount in milliliters, `0..400`
- `sunlight_adjustment`: `increase`, `decrease`, or `maintain`
- `nutrients`: `add` or `none`
- `energy_mode`: `low`, `normal`, or `high`

## 🎯 Tasks

### Easy Task

Maintain a single plant with stable weather and full starting resources.

### Medium Task

Manage two plants with limited water and energy under moderate weather variation.

### Hard Task

Optimize multi-plant growth under dynamic weather and strict resource constraints.

## 🧮 Grading System

Each task is scored between `0.0` and `1.0`:

- `1.0` means optimal, resource-efficient growth
- `0.5` means moderate success
- `0.0` means plant failure or severe resource misuse

Grading is deterministic and reproducible.

Implemented graders:
- `grade_easy(state)`
- `grade_medium(state)`
- `grade_hard(state)`

## ⚙️ Environment Design

- clean state reset at each episode
- structured observation and action spaces
- continuous reward shaping instead of sparse terminal-only reward
- deterministic transitions with fixed seeds
- clearly defined episode boundaries through step limits and failure conditions

Reward shaping combines:
- plant health improvement
- growth progress
- resource efficiency
- overuse penalties
- plant damage penalties

## ✅ OpenEnv Compliance

- typed `Observation`, `Action`, and `Reward` models
- `reset()`, `step()`, and `state()` implemented
- `openenv.yaml` included
- deterministic task variants: `easy`, `medium`, `hard`
- reproducible inference script
- Dockerfile included for deployment

## 📦 Project Structure

- `inference.py`: strict baseline inference runner
- `microfarm_env.py`: OpenEnv-compatible environment
- `env/models.py`: Pydantic models and shared environment types
- `env/grader.py`: deterministic task graders
- `openenv.yaml`: root environment manifest
- `config/openenv.yaml`: adapter-facing environment manifest
- `app.py`: Streamlit demo UI
- `Dockerfile`: deployment container

## 💡 Novelty

This environment introduces:
- multi-objective optimization between health and resource efficiency
- real-world sustainability constraints
- decision-based AI interaction instead of game-style actions
- explainable AI behavior through structured decisions and logged trajectories

It goes beyond traditional RL environments by simulating an operational agricultural decision system.

## Setup Instructions

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run the Streamlit interface:

```powershell
streamlit run app.py
```

Run strict inference:

```powershell
python inference.py --max-steps 8
```

## Inference Format

The inference runner emits:

```text
[START] task=<task> env=urban_micro_farm model=<model>
[STEP] step=<n> action=<json-string> reward=<float> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> score=<0-1> rewards=<r1,r2,...>
```

Example:

```text
[START] task=easy env=urban_micro_farm model=gpt-4o-mini
[STEP] step=1 action={"energy_mode":"low","nutrients":"none","sunlight_adjustment":"maintain","water_ml":0.0} reward=0.52 done=false error=null
[END] success=true steps=8 score=0.78 rewards=0.52,0.49,0.49
```

## Baseline Scores

The baseline inference prints one normalized score per task in `[0.0, 1.0]`.

Scoring depends on:
- final plant health
- growth quality
- resource efficiency
- multi-plant stability on harder tasks

## Docker / HF Space

The project includes a minimal deployment container:
- Python 3.10
- installs from `requirements.txt`
- runs `streamlit run app.py --server.port=7860 --server.address=0.0.0.0`

This is designed to run on CPU-only infrastructure and fit hackathon limits.
