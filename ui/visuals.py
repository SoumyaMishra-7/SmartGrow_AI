from __future__ import annotations


def mini_bar(value: float, width: int = 20) -> str:
    filled = max(0, min(width, int(round(value * width))))
    return "[" + "#" * filled + "." * (width - filled) + "]"
