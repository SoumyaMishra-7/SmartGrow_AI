from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import load_runtime_config
from env.resource import ACTIONS
from env.smart_env import SmartGrowEnv
from tasks.scenarios import DEFAULT_SCENARIO
from training.config import TrainingConfig
from training.train_dqn import run_training


SCENARIO_OPTIONS = {
    "balanced": "Balanced day",
    "hot_dry": "Hot and dry day",
    "stormy": "Stormy day",
}

RUNTIME_CONFIG = load_runtime_config()


def _state_defaults() -> None:
    if "env" not in st.session_state:
        st.session_state.env = SmartGrowEnv()
        st.session_state.observation, _ = st.session_state.env.reset()
        st.session_state.history = []
        st.session_state.total_reward = 0.0
        st.session_state.agent = None
        st.session_state.auto_run = False
        st.session_state.config = {
            "scenario_name": RUNTIME_CONFIG.get("scenario", DEFAULT_SCENARIO),
            "episodes": 80,
            "max_days": int(RUNTIME_CONFIG.get("max_days", 30)),
            "seed": int(RUNTIME_CONFIG.get("seed", 7)),
            "learning_rate": float(RUNTIME_CONFIG.get("learning_rate", 0.18)),
            "discount": float(RUNTIME_CONFIG.get("discount", 0.94)),
            "epsilon": float(RUNTIME_CONFIG.get("epsilon", 1.0)),
            "epsilon_decay": float(RUNTIME_CONFIG.get("epsilon_decay", 0.985)),
            "epsilon_min": float(RUNTIME_CONFIG.get("epsilon_min", 0.05)),
        }


def _reset_simulation() -> None:
    config = st.session_state.config
    st.session_state.env = SmartGrowEnv(
        scenario_name=config["scenario_name"],
        max_days=config["max_days"],
        seed=config["seed"],
    )
    st.session_state.observation, _ = st.session_state.env.reset(seed=config["seed"])
    st.session_state.history = []
    st.session_state.total_reward = 0.0
    st.session_state.auto_run = False


def _train_agent() -> None:
    config = st.session_state.config
    training = run_training(
        TrainingConfig(
            scenario_name=config["scenario_name"],
            episodes=config["episodes"],
            max_days=config["max_days"],
            learning_rate=config["learning_rate"],
            discount=config["discount"],
            epsilon=config["epsilon"],
            epsilon_decay=config["epsilon_decay"],
            epsilon_min=config["epsilon_min"],
            seed=config["seed"],
            log_interval=max(1, config["episodes"] // 4),
            save_plot=False,
        )
    )
    st.session_state.agent = training.agent
    st.session_state.training_summary = training.metrics.summary()


def _step_once() -> None:
    if st.session_state.agent is None:
        _train_agent()

    action_id = st.session_state.agent.act(st.session_state.observation, training=False)
    step = st.session_state.env.step(action_id)
    st.session_state.observation = step.observation
    st.session_state.total_reward += step.reward
    row = {
        "day": step.info["day"],
        "reward": step.reward,
        "total_reward": st.session_state.total_reward,
        "growth": step.info["growth"],
        "health": step.info["health"],
        "water": step.info["soil_moisture"],
        "nutrients": step.info["nutrients"],
        "energy": step.info["energy_reserve"],
        "temperature": step.info["temperature"],
        "action": step.info["action"],
    }
    st.session_state.history.append(row)
    if step.terminated or step.truncated:
        st.session_state.auto_run = False


def _history_frame() -> pd.DataFrame:
    history = st.session_state.history
    if not history:
        return pd.DataFrame(
            [{"day": 0, "reward": 0.0, "total_reward": 0.0, "growth": 0.1, "health": 0.95, "water": 0.55, "nutrients": 0.55, "energy": 1.0}]
        )
    return pd.DataFrame(history)


def _plant_stage(growth: float, health: float) -> str:
    if health < 0.35:
        return "Plant status: stressed"
    if growth < 0.35:
        return "Plant status: seedling"
    if growth < 0.75:
        return "Plant status: vegetative"
    if growth < 1.0:
        return "Plant status: flowering"
    return "Plant status: harvest-ready"


def _weather_indicator(temperature: float, humidity: float, light: float) -> str:
    if temperature > 29:
        return "Hot"
    if light < 0.4:
        return "Cloudy"
    if humidity > 0.75:
        return "Humid"
    return "Sunny"


def _care_level(value: float) -> str:
    if value < 0.3:
        return "Low"
    if value < 0.7:
        return "Okay"
    return "Good"


def _format_percent(value: float) -> str:
    return f"{int(round(value * 100))}%"


def _friendly_action_label(raw_label: str) -> str:
    labels = {
        "observe": "Observe only",
        "water_light": "Give a little water",
        "water_and_feed": "Water and feed",
        "deep_water": "Give extra water",
        "feed_boost": "Add nutrients",
        "shade_canopy": "Add shade",
        "ventilate": "Improve airflow",
        "climate_control": "Balance the climate",
    }
    return labels.get(raw_label, raw_label.replace("_", " ").title())


def _status_message(state, env: SmartGrowEnv) -> tuple[str, str]:
    if state.health <= 0.15:
        return ("error", "The plant is failing. Reset or retrain to try a safer strategy.")
    if state.growth >= 1.0:
        return ("success", "The plant is ready to harvest.")
    if state.day >= env.max_days:
        return ("info", "This run has reached the final day.")
    if state.soil_moisture < 0.3:
        return ("warning", "The soil is getting dry.")
    if state.nutrients < 0.3:
        return ("warning", "Nutrients are getting low.")
    if state.health < 0.4:
        return ("warning", "The plant looks stressed.")
    return ("success", "The plant is stable right now.")


def _history_table(history: pd.DataFrame) -> pd.DataFrame:
    table = history.tail(12).copy()
    return table.rename(
        columns={
            "day": "Day",
            "reward": "Reward this step",
            "total_reward": "Total reward",
            "growth": "Growth",
            "health": "Health",
            "water": "Water",
            "nutrients": "Nutrients",
            "energy": "Energy",
            "temperature": "Temperature",
            "action": "Action taken",
        }
    )


def _training_table(summary: dict[str, float]) -> pd.DataFrame:
    rows = [
        ("Average reward", summary.get("reward")),
        ("Average loss", summary.get("loss")),
        ("Average growth", summary.get("growth")),
        ("Average health", summary.get("health")),
    ]
    return pd.DataFrame(rows, columns=["Training metric", "Value"])


def main() -> None:
    st.set_page_config(page_title="SmartGrow AI", layout="wide")
    _state_defaults()

    with st.sidebar:
        st.title("SmartGrow AI")
        st.markdown("Use this panel to set up the plant simulation.")

        scenario_keys = list(SCENARIO_OPTIONS)
        selected_index = scenario_keys.index(st.session_state.config["scenario_name"])
        scenario_name = st.selectbox(
            "Choose the weather style",
            scenario_keys,
            index=selected_index,
            format_func=lambda key: SCENARIO_OPTIONS[key],
            help="This changes how easy or difficult the growing conditions are.",
        )
        episodes = st.slider(
            "Practice rounds for the AI",
            min_value=20,
            max_value=300,
            value=st.session_state.config["episodes"],
            step=10,
            help="Higher values usually make the AI more reliable, but training takes longer.",
        )
        max_days = st.slider(
            "Days to simulate",
            min_value=10,
            max_value=60,
            value=st.session_state.config["max_days"],
            step=5,
            help="This controls how long one plant run will last.",
        )
        seed = st.number_input(
            "Repeatable run number",
            min_value=1,
            max_value=9999,
            value=st.session_state.config["seed"],
            help="Keep this the same if you want the same random setup again.",
        )
        st.session_state.config = {
            "scenario_name": scenario_name,
            "episodes": episodes,
            "max_days": max_days,
            "seed": int(seed),
            "learning_rate": st.session_state.config["learning_rate"],
            "discount": st.session_state.config["discount"],
            "epsilon": st.session_state.config["epsilon"],
            "epsilon_decay": st.session_state.config["epsilon_decay"],
            "epsilon_min": st.session_state.config["epsilon_min"],
        }
        if st.button("Start a new smart plan", use_container_width=True):
            _train_agent()
            _reset_simulation()
        st.caption("After training, the app will choose the next plant-care action for you.")

    st.title("SmartGrow Plant Care Dashboard")
    st.write("This page shows how the AI is caring for the plant. Use the buttons below to move one day at a time or let it run by itself.")

    with st.container(border=True):
        st.subheader("How to use this page")
        st.markdown(
            "1. Pick a weather style and training strength in the left panel.\n"
            "2. Click **Start a new smart plan**.\n"
            "3. Click **Next day** to move one step at a time, or **Run automatically** to keep going.\n"
            "4. Watch the plant health and growth cards to see if the plan is working."
        )

    control_a, control_b, control_c = st.columns(3)
    with control_a:
        if st.button("Start over", use_container_width=True):
            _reset_simulation()
    with control_b:
        if st.button("Next day", use_container_width=True):
            _step_once()
    with control_c:
        auto_run_label = "Stop automatic run" if st.session_state.auto_run else "Run automatically"
        if st.button(auto_run_label, use_container_width=True):
            st.session_state.auto_run = not st.session_state.auto_run

    env = st.session_state.env
    state = env.state
    history = _history_frame()
    status_kind, status_text = _status_message(state, env)

    if status_kind == "error":
        st.error(status_text)
    elif status_kind == "warning":
        st.warning(status_text)
    elif status_kind == "success":
        st.success(status_text)
    else:
        st.info(status_text)

    metric_a, metric_b, metric_c, metric_d, metric_e = st.columns(5)
    metric_a.metric("Plant growth", _format_percent(state.growth), help="Higher is better. 100% means harvest-ready.")
    metric_b.metric("Plant health", _format_percent(state.health), help="Higher means the plant is in better condition.")
    metric_c.metric("Soil water", f"{_care_level(state.soil_moisture)} ({_format_percent(state.soil_moisture)})")
    metric_d.metric("Plant nutrients", f"{_care_level(state.nutrients)} ({_format_percent(state.nutrients)})")
    metric_e.metric("Day", f"{state.day} of {env.max_days}")

    left, right = st.columns([2, 1])
    with left:
        st.subheader("Progress over time")
        st.caption("These charts show whether the plant is improving as days pass.")
        st.line_chart(history.set_index("day")[["growth", "health", "water", "nutrients", "energy"]])
        with st.expander("Show advanced score chart"):
            st.line_chart(history.set_index("day")[["reward", "total_reward"]])
    with right:
        st.subheader("Current snapshot")
        st.progress(min(max(state.growth, 0.0), 1.0), text=_plant_stage(state.growth, state.health))
        st.write(f"Weather today: **{_weather_indicator(state.temperature, state.humidity, state.light)}**")
        st.write(f"Last AI action: **{_friendly_action_label(env.last_action)}**")
        st.write(f"Energy reserve: **{_care_level(state.energy_reserve)} ({_format_percent(state.energy_reserve)})**")
        st.write(f"Temperature: **{state.temperature:.1f} C**")
        st.write("Possible care actions")
        for action_id, effect in ACTIONS.items():
            st.caption(f"{action_id}: {_friendly_action_label(effect.label)}")

    st.subheader("Recent activity")
    st.caption("This table shows the latest decisions and how the plant changed.")
    st.dataframe(_history_table(history), use_container_width=True)

    if "training_summary" in st.session_state:
        with st.expander("Show training details"):
            st.dataframe(_training_table(st.session_state.training_summary), use_container_width=True)
            st.json(st.session_state.training_summary)

    if st.session_state.auto_run and state.day < env.max_days and state.health > 0.15 and state.growth < 1.0:
        _step_once()
        time.sleep(0.35)
        st.rerun()


if __name__ == "__main__":
    main()
