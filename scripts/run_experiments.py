"""Train, evaluate and record both MountainCar agents."""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from mountain_car.agents import DQNAgent, QLearningAgent


def evaluate(agent, episodes: int, seed: int) -> dict[str, float | int]:
    env = gym.make("MountainCar-v0")
    rewards: list[float] = []
    successes = 0
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        total = 0.0
        while True:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            if terminated or truncated:
                successes += int(terminated)
                rewards.append(total)
                break
    env.close()
    return {
        "episodes": episodes,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "best_reward": float(np.max(rewards)),
        "worst_reward": float(np.min(rewards)),
        "successes": successes,
        "success_rate": successes / episodes,
    }


def write_history(path: Path, rewards: list[float], window: int = 100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(("episode", "reward", f"moving_average_{window}"))
        for index, reward in enumerate(rewards, 1):
            start = max(0, index - window)
            writer.writerow((index, reward, float(np.mean(rewards[start:index]))))


def training_summary(rewards: list[float], window: int = 100) -> dict[str, float | int]:
    if not rewards:
        raise ValueError("Training must contain at least one episode")
    window = min(window, len(rewards))
    moving_averages = [
        float(np.mean(rewards[index - window : index]))
        for index in range(window, len(rewards) + 1)
    ]
    best_index = int(np.argmax(moving_averages))
    return {
        f"last_{window}_training_mean": float(np.mean(rewards[-window:])),
        f"best_{window}_training_mean": moving_averages[best_index],
        f"best_{window}_training_episode": best_index + window,
    }


def make_svg(path: Path, histories: dict[str, list[float]]) -> None:
    width, height = 960, 520
    left, right, top, bottom = 75, 25, 35, 65
    plot_w, plot_h = width - left - right, height - top - bottom
    colors = {"Q-Learning": "#2563eb", "DQN": "#dc2626"}
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#222"/>',
        '<text x="480" y="505" text-anchor="middle" font-family="sans-serif">Progreso del entrenamiento (%)</text>',
        '<text x="18" y="250" text-anchor="middle" transform="rotate(-90 18 250)" font-family="sans-serif">Recompensa media (100 episodios)</text>',
    ]
    for reward in (-200, -175, -150, -125, -100):
        y = top + (-100 - reward) / 100 * plot_h
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#ddd"/>')
        lines.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{reward}</text>')
    for name, rewards in histories.items():
        window = 100
        values = [float(np.mean(rewards[max(0, i - window + 1) : i + 1])) for i in range(len(rewards))]
        stride = max(1, len(values) // 500)
        points = []
        for i in range(0, len(values), stride):
            x = left + (i / max(1, len(values) - 1)) * plot_w
            y = top + (-100 - np.clip(values[i], -200, -100)) / 100 * plot_h
            points.append(f"{x:.1f},{y:.1f}")
        lines.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[name]}" stroke-width="2"/>')
    for index, name in enumerate(histories):
        x = left + 20 + index * 190
        lines.append(f'<line x1="{x}" y1="20" x2="{x + 30}" y2="20" stroke="{colors[name]}" stroke-width="3"/>')
        lines.append(f'<text x="{x + 38}" y="24" font-family="sans-serif" font-size="13">{name}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-episodes", type=int, default=20_000)
    parser.add_argument("--dqn-episodes", type=int, default=2_500)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output = Path("results")
    output.mkdir(exist_ok=True)

    agents = {
        "qlearning": QLearningAgent("MountainCar-v0"),
        "dqn": DQNAgent("MountainCar-v0"),
    }
    episode_counts = {"qlearning": args.q_episodes, "dqn": args.dqn_episodes}
    histories: dict[str, list[float]] = {}
    summary: dict[str, dict] = {}
    for name, agent in agents.items():
        start = time.perf_counter()
        history = agent.train(episode_counts[name])
        elapsed = time.perf_counter() - start
        agent.save(Path("saves") / ("qlearning_mountaincar.pkl" if name == "qlearning" else "dqn_mountaincar.pt"))
        write_history(output / f"{name}_training.csv", history)
        histories["Q-Learning" if name == "qlearning" else "DQN"] = history
        summary[name] = {
            "training_episodes": episode_counts[name],
            "training_seconds": elapsed,
            **training_summary(history),
            "evaluation": evaluate(agent, args.eval_episodes, args.seed + 10_000),
        }

    summary["seed"] = args.seed
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    make_svg(output / "training_curves.svg", histories)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
