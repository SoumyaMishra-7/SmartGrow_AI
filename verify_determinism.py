from __future__ import annotations

from agent.run_baseline import collect_scores


def main() -> None:
    first = collect_scores()
    second = collect_scores()

    if first != second:
        raise RuntimeError(
            "Determinism verification failed: baseline outputs differ between runs. "
            f"run1={first} run2={second}"
        )

    print("Determinism verification passed: repeated baseline runs are identical.")
    print(first)


if __name__ == "__main__":
    main()
