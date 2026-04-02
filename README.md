# SmartGrow AI

SmartGrow AI is a lightweight plant-care simulator with a tabular reinforcement-learning agent, a CLI training loop, and a Streamlit dashboard for visualizing plant growth over time.

## Project Layout

- `env/`: simulation state, weather, reward, and environment adapter code
- `agent/`: tabular DQN-style agent, policy, replay buffer, and baseline agent
- `training/`: training config, training loop, and evaluation
- `tasks/`: scenario and benchmark task definitions
- `ui/`: terminal report helpers and Streamlit dashboard
- `config/`: runtime defaults loaded by the CLI and dashboard
- `config/openenv.yaml`: explicit OpenEnv spec artifact
- `tests/`: regression tests for env, reward, config loading, and adapter behavior

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Run Training From CLI

```powershell
.\.venv\Scripts\python main.py
```

You can override runtime parameters:

```powershell
.\.venv\Scripts\python main.py --scenario hot_dry --episodes 120 --days 40 --seed 11
```

## Visualize In Streamlit

```powershell
.\.venv\Scripts\python -m streamlit run ui\streamlit_app.py
```

The dashboard lets you train an agent, step the simulation day by day, and watch growth, health, water, nutrient, and reward trends update live.

## Baseline Benchmark

```powershell
.\.venv\Scripts\python baseline_inference.py
```

## Tests

```powershell
.\.venv\Scripts\python -m pytest -q
```

## Docker

Build the image:

```powershell
docker build -t smartgrow-ai .
```

Run the default CLI entrypoint:

```powershell
docker run --rm smartgrow-ai
```
