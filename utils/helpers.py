from __future__ import annotations


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def moving_average(values: list[float], window: int = 10) -> list[float]:
    if not values:
        return []
    output: list[float] = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        sample = values[start : index + 1]
        output.append(sum(sample) / len(sample))
    return output
