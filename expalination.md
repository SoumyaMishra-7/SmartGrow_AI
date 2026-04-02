# SmartGrow AI Project Explanation

## 1. What this project is

SmartGrow AI is a beginner-friendly reinforcement learning project that simulates a real-world plant care task.

Instead of playing a game, the AI is trying to solve a practical control problem:

- how much to water a plant
- when to add nutrients
- when to add shade
- when to improve airflow
- how to keep the plant healthy across changing weather conditions

The project includes:

- a plant growth simulation environment
- an AI agent that learns by trial and error
- a baseline rule-based agent
- three graded tasks from easy to hard
- a command-line training pipeline
- a Streamlit dashboard for visualization
- OpenEnv-style adapter methods and typed models

This makes it suitable as a small but complete AI systems project.

---

## 2. Main idea in simple words

Imagine a smart greenhouse assistant.

Every day, the plant has a condition:

- soil moisture
- nutrients
- temperature
- humidity
- light
- growth
- health
- energy reserve

The AI looks at the current situation and chooses one action, such as:

- observe only
- light watering
- water and feed
- deep watering
- nutrient boost
- shade canopy
- ventilation
- climate control

After that:

1. the weather changes
2. the action affects the plant
3. the plant grows or becomes stressed
4. the system gives the AI a reward

The AI repeats this many times and slowly learns which actions are better.

---

## 3. What type of AI is used here

This project uses a lightweight reinforcement learning approach.

More specifically:

- the file names call it a DQN-style agent
- but the actual implementation is a tabular Q-learning agent
- that means it stores action values in a table instead of using a deep neural network

This is actually a good beginner design because:

- it is easier to understand
- it is fast to run
- it does not need PyTorch or TensorFlow
- it still demonstrates how reinforcement learning works

---

## 4. Tech stack

### Core language

- Python

### Libraries used

- `streamlit` for the dashboard
- `pandas` for tables/history visualization
- `matplotlib` for saving reward plots
- `pytest` for testing
- `json` and standard library modules for config/spec loading

### Development/runtime style

- modular Python package structure
- dataclasses for typed state/config/result models
- command-line interface with `argparse`
- Streamlit for interactive visualization
- Docker support for containerized build/run

---

## 5. Folder-by-folder explanation

## Root folder

This is the main project directory. It contains entry scripts, configs, tests, and modular folders.

### Root files

#### `main.py`

Main CLI training entrypoint.

What it does:

- reads runtime defaults from config
- accepts command-line arguments
- builds a `TrainingConfig`
- trains the learning agent
- evaluates the agent
- prints a readable training report

Use this when you want to train and test the AI from terminal.

#### `baseline_inference.py`

Rule-based baseline benchmark script.

What it does:

- runs the deterministic baseline agent on the official tasks
- calculates outcome metrics
- grades each task
- prints reproducible scores

Why it matters:

- it satisfies the baseline inference requirement
- it gives a non-learning reference score to compare against

#### `requirements.txt`

Standard dependency file for setup and Docker builds.

Use:

```powershell
pip install -r requirements.txt
```

#### `requirements.py`

Python list of dependencies.

This is not the standard installation file. The real setup file is `requirements.txt`.

#### `README.md`

Short project guide with setup, run, test, visualization, and Docker commands.

#### `pytest.ini`

Pytest configuration file.

What it does:

- forces tests to be collected only from the `tests/` folder
- disables the cache plugin to avoid local Windows permission issues

#### `Dockerfile`

Container definition for automated build.

What it does:

- uses Python 3.11 slim image
- installs dependencies
- copies the project
- runs `python main.py` by default

#### `.dockerignore`

Prevents unnecessary or problematic files from being copied into Docker build context.

#### `.gitignore`

Tells Git to ignore generated files such as:

- Python cache files
- virtual environment
- plot outputs
- notebook and pytest cache artifacts

#### `expalination.md`

This file. It explains the project in detail.

---

## `agent/`

This folder contains the AI decision logic.

### `agent/__init__.py`

Exports the main agent class so it can be imported cleanly.

### `agent/dqn_agent.py`

Core learning agent.

Important note:

- the name says DQN
- but internally it is a tabular Q-learning agent

What it does:

- stores Q-values for `(state, action)` pairs
- chooses actions using epsilon-greedy exploration
- updates the Q-table after each step
- slowly reduces exploration over time

This is the heart of the learning process.

### `agent/policy.py`

Contains `epsilon_greedy`.

What it does:

- sometimes picks a random action to explore
- otherwise picks the action with the highest known value

This balances:

- exploration: trying new things
- exploitation: using what it already learned

### `agent/replay_buffer.py`

Stores transitions.

A transition means:

- current observation
- selected action
- received reward
- next observation
- whether the episode ended

In this project the buffer is lightweight and mostly used as a clean RL structure component.

### `agent/baseline_agent.py`

Rule-based non-learning agent.

What it does:

- uses simple if-else logic
- waters when moisture is low
- feeds when nutrients are low
- uses climate actions when temperature/light are risky

This gives reproducible baseline scores.

### `agent/agent.py`

Legacy or alternate agent-related file from earlier development.

It is not the main learning path used by the current training pipeline.

---

## `env/`

This folder is the simulation engine. It represents the world where the AI acts.

### `env/__init__.py`

Exports `SmartGrowEnv`.

### `env/smart_env.py`

Main simulation environment.

This is the most important environment file.

It provides:

- `reset()`
- `step()`
- current state management
- action effects
- reward calculation
- terminal conditions
- render string for logging

This is the environment the AI learns inside.

### `env/state.py`

Defines `GardenState`.

It stores plant and environment values such as:

- day
- soil moisture
- nutrients
- temperature
- humidity
- light
- growth
- health
- water tank
- nutrient tank
- energy reserve

It also converts raw values into bucketed observations for the agent.

Why bucketing matters:

- the Q-learning table cannot handle infinite continuous values directly
- so values are grouped into bins
- this makes the state small enough for tabular learning

### `env/resource.py`

Defines all available actions and their effects.

Examples:

- `observe`
- `water_light`
- `water_and_feed`
- `deep_water`
- `feed_boost`
- `shade_canopy`
- `ventilate`
- `climate_control`

This file tells the environment what each action changes.

### `env/weather.py`

Generates day-to-day weather.

It simulates:

- temperature
- humidity
- light
- rainfall

based on the chosen scenario.

### `env/plant.py`

Updates the plant’s growth and health.

It compares the current conditions to the ideal conditions of the scenario and computes:

- stress
- growth gain
- health change

This is where the “biology-like” part of the simulator lives.

### `env/reward.py`

Defines the reward function.

This is extremely important in reinforcement learning.

The reward includes:

- growth improvement reward
- health improvement reward
- stability reward
- stress penalty
- resource penalty
- milestone rewards
- terminal success bonus
- failure penalty

This is a shaped reward function, which means the AI gets partial progress signals, not just final success/failure.

### `env/models.py`

Contains typed dataclasses used across the environment and OpenEnv adapter.

Includes:

- `ActionEffect`
- `StepResult`
- `EpisodeTrace`
- `OpenEnvActionModel`
- `OpenEnvStateModel`
- `OpenEnvResetModel`
- `OpenEnvStepModel`

This helps make the project cleaner and more structured.

### `env/openenv.py`

OpenEnv-style adapter layer.

What it provides:

- `reset()`
- `step()`
- `state()`
- `spec()`
- `render()`

This file makes the environment easier to integrate with a standard interface style.

### `env/env.py`

Compatibility wrapper that re-exports `SmartGrowEnv`.

### `env/openenv.py`

Already explained above. This is the OpenEnv interface layer.

---

## `tasks/`

This folder defines official tasks and grading logic.

### `tasks/__init__.py`

Exports basic scenario symbols.

### `tasks/scenarios.py`

Defines the environment scenarios:

- `balanced`
- `hot_dry`
- `stormy`

Each scenario contains:

- base temperature
- humidity
- light
- rain behavior
- ideal growing values

These scenarios change how hard the learning problem is.

### `tasks/tasks.py`

Defines task specs and graders.

Current tasks:

- `easy_balanced_growth`
- `medium_heat_resilience`
- `hard_storm_recovery`

This file contains:

- task metadata
- difficulty labels
- success hints
- task lookup helpers
- grader functions

Each grader returns a score between `0.0` and `1.0`.

This satisfies the requirement for:

- at least 3 tasks
- increasing difficulty
- task graders

---

## `training/`

This folder contains the learning pipeline.

### `training/__init__.py`

Exports `TrainingConfig`.

### `training/config.py`

Defines the training settings dataclass.

Important settings include:

- scenario name
- number of episodes
- max days per episode
- learning rate
- discount factor
- epsilon
- epsilon decay
- minimum epsilon
- seed
- logging interval
- whether to save plots

### `training/train_dqn.py`

Runs the training loop.

What happens here:

1. create environment
2. create learning agent
3. repeat for many episodes
4. reset environment
5. choose actions
6. step environment
7. learn from reward
8. record metrics
9. decay epsilon
10. optionally save reward curve

Even though the file says DQN, this is tabular RL training.

### `training/evaluate.py`

Runs a trained agent without exploration.

What it does:

- resets the environment
- always chooses the best known action
- records total reward
- tracks final growth and health
- stores a trace of environment states

This is used after training to see how well the agent performs.

---

## `ui/`

This folder contains text and dashboard visualization.

### `ui/__init__.py`

Exports the terminal report printer.

### `ui/app.py`

Prints CLI training and evaluation summaries in a clean report format.

### `ui/components.py`

Small helpers for formatting text output:

- aligned stat lines
- titled sections

### `ui/visuals.py`

Contains `mini_bar`, a small ASCII progress bar used in CLI reports.

### `ui/streamlit_app.py`

Main web dashboard.

This is the visualization part of the project.

It allows the user to:

- choose scenario
- choose training strength
- start a new trained plan
- step one day at a time
- auto-run the simulation
- see plant metrics live
- view history tables
- view progress charts
- inspect training summaries

This is the best file to run if you want a visual explanation of what the AI is doing.

---

## `utils/`

Utility helpers used across the project.

### `utils/__init__.py`

Package marker / export support.

### `utils/helpers.py`

Contains:

- `clamp()` for limiting values
- `moving_average()` for smoothing metrics

These are used throughout the simulator and training reports.

### `utils/metrics.py`

Tracks training metrics like:

- rewards
- growth
- health
- loss

It also computes summary values.

### `utils/plotting.py`

Saves reward history to:

- CSV
- PNG plot if matplotlib is available

This creates learning curve outputs under `models/plots/`.

### `utils/logger.py`

Simple logger helper for readable console logging.

---

## `config/`

This folder stores runtime configuration and OpenEnv metadata.

### `config/__init__.py`

Exports the config loader.

### `config/loader.py`

Loads simple config values from YAML-like files.

Used by:

- `main.py`
- `ui/streamlit_app.py`

This reduces duplicated defaults.

### `config/env_config.yaml`

Environment runtime defaults, such as:

- default scenario
- max days
- seed

### `config/model_config.yaml`

Model/training runtime defaults, such as:

- learning rate
- discount
- epsilon values

### `config/openenv.yaml`

Explicit OpenEnv-style specification file.

This exists to satisfy the requirement that the project include:

- `openenv.yaml`

It describes:

- environment name
- description
- observation fields
- action count
- termination settings

---

## `tests/`

This folder contains automated checks.

### `tests/test_env.py`

Verifies that:

- the environment returns an 8-part observation
- `step()` returns proper result structure
- the environment terminates within the configured horizon

### `tests/test_agent.py`

Verifies that:

- the learning update changes Q-values
- the agent returns valid action indices

### `tests/test_reward.py`

Verifies that:

- better growth/health gives positive reward
- failure gives negative reward

### `tests/test_config_loader.py`

Checks that runtime config loading works correctly.

### `tests/test_openenv.py`

Checks that:

- `config/openenv.yaml` exists
- the OpenEnv adapter loads the spec correctly

---

## `models/`

This folder stores generated outputs.

### `models/plots/`

Contains training reward curves and CSV/plot artifacts.

Example outputs:

- reward history CSV
- reward curve image

These are generated during training when plot saving is enabled.

---

## `notebooks/`

Contains notebook-based experimentation.

### `notebooks/experiments.ipynb`

Used for exploratory work, manual testing, or demonstrations.

This is not required for the main run pipeline.

---

## 6. How the AI learns

This is the most important section for beginners.

### Step 1: The environment gives a state

The plant has a condition, such as:

- water level
- nutrients
- temperature
- health

These values are bucketed into a simpler observation.

Example idea:

- low moisture
- medium nutrients
- high temperature
- good health

This observation becomes the AI’s current state.

### Step 2: The agent picks an action

The agent checks its Q-table.

The Q-table stores:

- for this kind of state
- how good each possible action seems

At first, the agent knows nothing, so it explores randomly a lot.

Later, it starts trusting what it learned.

### Step 3: The environment reacts

The environment applies:

- weather changes
- action effects
- plant growth rules
- health updates

Then it returns:

- next state
- reward
- whether the episode ended

### Step 4: The agent updates its knowledge

The agent uses a Q-learning update:

- if an action led to a good future, increase its value
- if an action led to a bad future, reduce its value

Over many episodes, the table becomes smarter.

### Step 5: Exploration slowly reduces

The agent starts with high epsilon:

- more random actions

Then epsilon decays:

- fewer random actions
- more best-known actions

This is how the agent moves from trying things to acting more confidently.

---

## 7. Beginner explanation of the reward function

The reward tells the AI whether it is doing well.

This project does not wait until the very end to reward the agent.
That would make learning too hard.

Instead, it gives partial signals during progress.

### Positive signals

- plant growth improves
- plant health improves
- milestones are reached
- successful ending gives a big bonus

### Negative signals

- stress conditions
- poor moisture/nutrient balance
- resource waste
- plant failure

Why this is good:

- the AI learns faster
- the agent gets feedback even before final success
- it matches the requirement for meaningful shaped rewards

---

## 8. Beginner explanation of the tasks and graders

The project has 3 official tasks.

### 1. Easy

`easy_balanced_growth`

Goal:

- grow a healthy plant in balanced weather

### 2. Medium

`medium_heat_resilience`

Goal:

- survive hot and dry conditions while keeping the plant healthy

### 3. Hard

`hard_storm_recovery`

Goal:

- handle unstable stormy weather and still finish strong

### How grading works

Each task has a dedicated grader function.

The grader looks at outcome metrics such as:

- final growth
- final health
- average health
- resource stability
- days survived
- whether the task completed
- whether the run failed

Then it returns a score between `0.0` and `1.0`.

This is useful because:

- different tasks can emphasize different skills
- the system can compare agents in a structured way

---

## 9. How the project runs from start to finish

When you run `main.py`, the flow is:

1. load config values
2. create training settings
3. create environment
4. create agent
5. run many episodes
6. update Q-table after each step
7. collect metrics
8. evaluate trained agent
9. print summary

When you run `ui/streamlit_app.py`, the flow is:

1. load config values
2. show sidebar controls
3. train agent when requested
4. reset simulation
5. step through plant care interactively
6. show charts and recent history

When you run `baseline_inference.py`, the flow is:

1. pick each official task
2. run the rule-based baseline agent
3. calculate outcome
4. run the task grader
5. print scores

---

## 10. Project requirements mapping

Here is how the project matches the common submission requirements.

### Real-world task

Satisfied.

Why:

- this is a plant care / greenhouse management simulation
- it is not a game or toy puzzle

### Full OpenEnv spec

Satisfied in this repo structure.

Present:

- typed models in `env/models.py`
- `reset()/step()/state()` in `env/openenv.py`
- explicit `config/openenv.yaml`

### 3+ tasks with graders

Satisfied.

Present in `tasks/tasks.py`:

- easy
- medium
- hard

Each returns scores in `0.0` to `1.0`.

### Meaningful reward function

Satisfied.

Present in `env/reward.py`:

- growth signals
- health signals
- milestone rewards
- penalties
- terminal/failure logic

### Baseline inference with reproducible scores

Satisfied.

Present in `baseline_inference.py`.

---

## 11. How to run the project

## Step 1: Create and activate virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

## Step 2: Install dependencies

```powershell
pip install -r requirements.txt
```

## Step 3: Run the CLI training flow

```powershell
.\.venv\Scripts\python main.py
```

Example with custom parameters:

```powershell
.\.venv\Scripts\python main.py --scenario hot_dry --episodes 120 --days 40 --seed 11
```

## Step 4: Run the visualization dashboard

```powershell
.\.venv\Scripts\python -m streamlit run ui\streamlit_app.py
```

What you will see:

- training controls in sidebar
- live plant metrics
- growth and health charts
- recent activity table
- training summary details

## Step 5: Run baseline benchmark

```powershell
.\.venv\Scripts\python baseline_inference.py
```

## Step 6: Run tests

```powershell
.\.venv\Scripts\python -m pytest -q
```

---

## 12. How to run with Docker

Build image:

```powershell
docker build -t smartgrow-ai .
```

Run container:

```powershell
docker run --rm smartgrow-ai
```

Note:

- Docker Desktop or Docker Engine must be running

---

## 13. What to open if you are a beginner

If you are new and want to understand the project quickly, read files in this order:

1. `README.md`
2. `main.py`
3. `training/train_dqn.py`
4. `agent/dqn_agent.py`
5. `env/smart_env.py`
6. `env/reward.py`
7. `tasks/tasks.py`
8. `ui/streamlit_app.py`

This order helps you move from:

- project overview
- to training flow
- to learning logic
- to environment logic
- to evaluation and visualization

---

## 14. Final summary

SmartGrow AI is a modular reinforcement learning project that teaches an AI to care for a plant under changing conditions.

It is a good beginner project because it shows:

- how an environment is built
- how an agent learns from rewards
- how scenarios and tasks are structured
- how scoring/grading works
- how to test a project
- how to visualize behavior

In one sentence:

This project is a complete small AI pipeline where a tabular reinforcement learning agent learns plant care decisions in a simulated real-world environment and can be trained, evaluated, graded, tested, visualized, and containerized.
