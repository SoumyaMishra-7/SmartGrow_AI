---
title: Urban Micro-Farm AI Assistant
sdk: streamlit
app_file: app.py
---

# Urban Micro-Farm AI Assistant

## Problem Description

Urban micro-farms operate under tight water, nutrient, and energy constraints. This project models a deterministic control environment where an agent must keep plants healthy, manage climate adjustments, and preserve limited resources across multiple task difficulties.

## Action Space

Each step accepts a JSON action shaped like:

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

## Observation Space

The OpenEnv observation returned by `MicroFarmEnv.reset()` and `MicroFarmEnv.step()` is:

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

## Tasks

- `easy`: single plant, stable weather, full starting resources
- `medium`: 2 plants, hotter conditions, moderate variation
- `hard`: 4 plants, dynamic weather rotation, tighter resources

## Setup

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run the Streamlit demo:

```powershell
streamlit run app.py
```

Run strict inference:

```powershell
python inference.py --max-steps 8
```

## Baseline Score

The baseline inference prints one `[END]` line per task with normalized `score=0.00..1.00`. Scores come from:

- `grade_easy(state)`
- `grade_medium(state)`
- `grade_hard(state)`

These graders use final environment state, plant health, plant growth, and remaining resources.
