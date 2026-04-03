from __future__ import annotations


def stat_line(label: str, value: object) -> str:
    return f"{label:<18}: {value}"


def section(title: str, lines: list[str]) -> str:
    border = "=" * len(title)
    return "\n".join([title, border, *lines])
