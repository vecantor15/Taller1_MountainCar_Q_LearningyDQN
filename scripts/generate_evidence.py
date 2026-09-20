"""Regenerate the individual evidence figures from recorded experiment files.

Usage:
    uv run --with matplotlib python scripts/generate_evidence.py

The script reads the CSV histories and summary.json already produced by
scripts/run_experiments.py. It does not retrain either agent.
"""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = Path("results")
WINDOW = 100


def read_rewards(path: Path) -> list[float]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return [float(row["reward"]) for row in csv.DictReader(file)]


def moving_average(rewards: list[float], window: int = WINDOW) -> tuple[np.ndarray, np.ndarray]:
    if len(rewards) < window:
        raise ValueError(f"Se requieren al menos {window} episodios")
    values = np.convolve(np.asarray(rewards, dtype=float), np.ones(window) / window, mode="valid")
    episodes = np.arange(window, len(rewards) + 1)
    return episodes, values


def plot_agent(agent_key: str, title: str, output_name: str, summary: dict) -> None:
    rewards = read_rewards(RESULTS_DIR / f"{agent_key}_training.csv")
    episodes, averages = moving_average(rewards)
    best_idx = int(np.argmax(averages))
    best_episode = int(episodes[best_idx])
    best_average = float(averages[best_idx])
    evaluation = summary[agent_key]["evaluation"]

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(episodes, averages, linewidth=1.5, label="Media móvil de 100 episodios")
    ax.axvline(best_episode, linestyle="--", linewidth=1.0, label=f"Mejor media: {best_average:.2f}")
    ax.axhline(
        float(evaluation["mean_reward"]),
        linestyle=":",
        linewidth=1.0,
        label=f"Media evaluación: {float(evaluation['mean_reward']):.2f}",
    )
    ax.set_title(title)
    ax.set_xlabel("Episodio")
    ax.set_ylabel("Recompensa media")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    # Reserve a dedicated strip below the axes so metrics never overlap the curve.
    fig.subplots_adjust(bottom=0.23)
    fig.text(
        0.5,
        0.055,
        (
            f"Mejor media móvil: {best_average:.2f} (episodio {best_episode})   |   "
            f"Evaluación: {float(evaluation['mean_reward']):.2f} ± {float(evaluation['std_reward']):.2f}   |   "
            f"Mejor episodio: {float(evaluation['best_reward']):.0f}   |   "
            f"Éxitos: {int(evaluation['successes'])}/{int(evaluation['episodes'])}"
        ),
        ha="center",
        fontsize=10,
    )
    fig.savefig(RESULTS_DIR / output_name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    summary = json.loads((RESULTS_DIR / "summary.json").read_text(encoding="utf-8"))
    plot_agent("qlearning", "Q-Learning — evolución del entrenamiento", "qlearning_evidence.png", summary)
    plot_agent("dqn", "DQN — evolución del entrenamiento", "dqn_evidence.png", summary)
    print("Evidencias regeneradas en results/qlearning_evidence.png y results/dqn_evidence.png")


if __name__ == "__main__":
    main()
