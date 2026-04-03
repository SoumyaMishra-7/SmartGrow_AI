from __future__ import annotations

from pathlib import Path

from utils.helpers import moving_average


def save_reward_curve(rewards: list[float], output_dir: str = "models/plots") -> str:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / "reward_curve.csv"
    png_path = output_path / "reward_curve.png"

    moving = moving_average(rewards, 10)
    with csv_path.open("w", encoding="utf-8") as file:
        file.write("episode,reward,moving_average\n")
        for index, reward in enumerate(rewards, start=1):
            file.write(f"{index},{reward},{moving[index - 1]}\n")

    try:
        import matplotlib.pyplot as plt  # type: ignore

        plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(rewards) + 1), rewards, label="reward", alpha=0.45)
        plt.plot(range(1, len(moving) + 1), moving, label="moving_avg_10", linewidth=2)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("SmartGrow Training Reward")
        plt.legend()
        plt.tight_layout()
        plt.savefig(png_path)
        plt.close()
        return str(png_path)
    except Exception:
        return str(csv_path)
