from __future__ import annotations

from pathlib import Path
from typing import Any


CONFIG_DIR = Path(__file__).resolve().parent
ENV_CONFIG_PATH = CONFIG_DIR / "env_config.yaml"
MODEL_CONFIG_PATH = CONFIG_DIR / "model_config.yaml"


def _parse_scalar(raw_value: str) -> Any:
    value = raw_value.strip()
    if not value:
        return ""
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if any(marker in value for marker in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("'\"")


def _parse_simple_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    parsed: dict[str, Any] = {}
    current_section: dict[str, Any] | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        indent = len(line) - len(line.lstrip(" "))
        key, _, raw_value = stripped.partition(":")
        key = key.strip()
        value = raw_value.strip()

        if indent == 0:
            if value:
                parsed[key] = _parse_scalar(value)
                current_section = None
            else:
                current_section = {}
                parsed[key] = current_section
        elif current_section is not None:
            current_section[key] = _parse_scalar(value)

    return parsed


def load_runtime_config() -> dict[str, Any]:
    env_config = _parse_simple_yaml(ENV_CONFIG_PATH)
    model_config = _parse_simple_yaml(MODEL_CONFIG_PATH)
    config = dict(env_config)
    config.update(model_config.get("agent", {}))
    return config
