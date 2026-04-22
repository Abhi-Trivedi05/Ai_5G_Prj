"""
main.py — Training, Evaluation & Visualization
================================================
This is the entry point of the project.  It:
  1.  Trains a Q-learning agent to turn small cells ON/OFF
  2.  Compares the agent against "Always ON" and "Random" baselines
  3.  Produces four publication-quality plots saved in results/
"""

import os
import csv
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

import config as cfg
from environment import SmallCellNetwork
from agent import QLearningAgent

# Ensure the results folder exists
os.makedirs("results", exist_ok=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1.  TRAINING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def train():
    """Train the Q-learning agent over many episodes."""
    env = SmallCellNetwork(seed=cfg.SEED)
    agent = QLearningAgent(env.num_cells, env.num_actions, seed=cfg.SEED)

    episode_rewards = []   # track total reward per episode
    episode_energies = []  # track total energy per episode

    print("=" * 60)
    print("  TRAINING STARTED")
    print(f"  Episodes: {cfg.NUM_EPISODES}   |   "
          f"Steps/episode: {cfg.STEPS_PER_EPISODE}")
    print("=" * 60)

    for ep in range(1, cfg.NUM_EPISODES + 1):
        state = env.reset()
        total_reward = 0.0
        total_energy = 0.0

        for step in range(cfg.STEPS_PER_EPISODE):
            # Agent selects an action
            action = agent.choose_action(state)

            # Environment processes the action
            next_state, reward, done, info = env.step(action)

            # Agent learns from the experience
            agent.learn(state, action, reward, next_state, done)

            total_reward += reward
            total_energy += info["energy"]
            state = next_state

        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        episode_energies.append(total_energy)

        # Print progress every 100 episodes
        if ep % 100 == 0 or ep == 1:
            avg_r = np.mean(episode_rewards[-100:])
            avg_e = np.mean(episode_energies[-100:])
            print(f"  Episode {ep:>4d}/{cfg.NUM_EPISODES}  |  "
                  f"Avg Reward: {avg_r:>8.1f}  |  "
                  f"Avg Energy: {avg_e:>7.1f} Wh   |  "
                  f"eps: {agent.epsilon:.3f}")

    agent.save()
    print("\n  Training complete!\n")
    return agent, episode_rewards, episode_energies


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2.  EVALUATION — run one day with a given policy
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def evaluate(agent, policy="rl", seed=42):
    """
    Simulate one full day (24 steps) and record per-hour metrics.

    Parameters
    ----------
    agent  : trained QLearningAgent  (used only when policy='rl')
    policy : 'rl'  — use trained agent
             'always_on' — keep all cells ON
             'random'    — random ON/OFF decisions

    Returns
    -------
    dict with keys: energy_per_hour, traffic_per_hour, status_per_hour
    """
    env = SmallCellNetwork(seed=seed)
    state = env.reset(seed=seed)

    energy_per_hour = []
    traffic_per_hour = []
    status_per_hour = []
    dropped_per_hour = []

    for step in range(cfg.STEPS_PER_EPISODE):
        if policy == "rl":
            action = agent.best_action(state)
        elif policy == "always_on":
            action = env.num_actions - 1  # all bits = 1 → all cells ON
        elif policy == "random":
            action = int(env.rng.integers(env.num_actions))
        else:
            raise ValueError(f"Unknown policy: {policy}")

        next_state, reward, done, info = env.step(action)

        energy_per_hour.append(info["energy"])
        traffic_per_hour.append(info["traffic"].copy())
        status_per_hour.append(info["cell_status"].copy())
        dropped_per_hour.append(info["dropped_traffic"])

        state = next_state

    return {
        "energy":   np.array(energy_per_hour),
        "traffic":  np.array(traffic_per_hour),
        "status":   np.array(status_per_hour),
        "dropped":  np.array(dropped_per_hour),
    }

def save_metrics_csv(result, policy_name, path):
    """Save per-hour metrics (energy/traffic/status/drops) to a CSV file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    headers = (
        ["hour", "energy_wh", "dropped_mbps"]
        + [f"traffic_cell_{i+1}_mbps" for i in range(cfg.NUM_SMALL_CELLS)]
        + [f"cell_{i+1}_on" for i in range(cfg.NUM_SMALL_CELLS)]
    )

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["policy", policy_name])
        w.writerow(["seed", cfg.SEED])
        w.writerow(headers)
        for hour in range(cfg.STEPS_PER_EPISODE):
            row = [
                hour,
                float(result["energy"][hour]),
                float(result["dropped"][hour]),
                *[float(x) for x in result["traffic"][hour]],
                *[int(x) for x in result["status"][hour]],
            ]
            w.writerow(row)


def save_summary_txt(rl_result, on_result, rand_result, path):
    """Write a short report-ready summary of key results."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    rl_total = float(rl_result["energy"].sum())
    on_total = float(on_result["energy"].sum())
    rand_total = float(rand_result["energy"].sum())

    saved_wh = on_total - rl_total
    saved_pct = (saved_wh / on_total * 100.0) if on_total > 0 else 0.0

    rl_dropped_total = float(rl_result["dropped"].sum())
    rand_dropped_total = float(rand_result["dropped"].sum())
    on_dropped_total = float(on_result["dropped"].sum())

    with open(path, "w", encoding="utf-8") as f:
        f.write("Energy-Efficient 5G RAN Optimization (Q-Learning)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"Seed: {cfg.SEED}\n")
        f.write(f"Small cells: {cfg.NUM_SMALL_CELLS}\n")
        f.write(f"Episodes: {cfg.NUM_EPISODES}\n")
        f.write(f"Steps/episode (hours): {cfg.STEPS_PER_EPISODE}\n\n")

        f.write("Total daily energy (Wh)\n")
        f.write(f"- Always ON : {on_total:.1f}\n")
        f.write(f"- Random    : {rand_total:.1f}\n")
        f.write(f"- RL Agent  : {rl_total:.1f}\n\n")

        f.write("Energy savings (RL vs Always ON)\n")
        f.write(f"- Saved: {saved_wh:.1f} Wh\n")
        f.write(f"- Saved: {saved_pct:.1f}%\n\n")

        f.write("Dropped traffic (Mbps total over 24 hours)\n")
        f.write(f"- Always ON : {on_dropped_total:.2f}\n")
        f.write(f"- Random    : {rand_dropped_total:.2f}\n")
        f.write(f"- RL Agent  : {rl_dropped_total:.2f}\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3.  PLOTTING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Use a clean style for all plots
plt.rcParams.update({
    "figure.facecolor": "#f8f9fa",
    "axes.facecolor":   "#ffffff",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "font.size":        11,
    "axes.titlesize":   14,
    "axes.labelsize":   12,
})


def plot_training_curve(episode_rewards):
    """Plot 1 — Reward convergence over training episodes."""
    fig, ax = plt.subplots(figsize=(10, 5))

    # Raw rewards (transparent)
    ax.plot(episode_rewards, alpha=0.2, color="#6c5ce7", linewidth=0.5)

    # Smoothed curve (moving average)
    window = 30
    if len(episode_rewards) >= window:
        smoothed = np.convolve(episode_rewards,
                               np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(episode_rewards)), smoothed,
                color="#6c5ce7", linewidth=2, label=f"Moving Avg ({window} ep)")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Training Reward Convergence")
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/1_training_curve.png", dpi=150)
    plt.close(fig)
    print("  -> Saved results/1_training_curve.png")


def plot_energy_comparison(rl_result, on_result, rand_result):
    """Plot 2 — Bar chart comparing total daily energy of 3 policies."""
    policies = ["Always ON", "Random", "RL Agent"]
    totals = [
        on_result["energy"].sum(),
        rand_result["energy"].sum(),
        rl_result["energy"].sum(),
    ]
    colors = ["#d63031", "#fdcb6e", "#00b894"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(policies, totals, color=colors, width=0.5, edgecolor="white",
                  linewidth=1.5)

    # Add value labels on bars
    for bar, val in zip(bars, totals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{val:.0f} Wh", ha="center", va="bottom", fontweight="bold")

    # Energy savings annotation
    saved = totals[0] - totals[2]
    pct = saved / totals[0] * 100
    ax.set_ylabel("Total Daily Energy Consumption (Wh)")
    ax.set_title(f"Energy Comparison - RL saves {pct:.1f}% vs Always-ON")
    fig.tight_layout()
    fig.savefig("results/2_energy_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  -> Saved results/2_energy_comparison.png  "
          f"(RL saved {saved:.0f} Wh = {pct:.1f}%)")


def plot_hourly_energy(rl_result, on_result):
    """Plot 3 — 24-hour energy profile: RL vs Always ON."""
    hours = np.arange(cfg.STEPS_PER_EPISODE)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(hours, on_result["energy"], alpha=0.15, color="#d63031")
    ax.plot(hours, on_result["energy"], "o-", color="#d63031",
            label="Always ON", linewidth=2, markersize=5)
    ax.fill_between(hours, rl_result["energy"], alpha=0.15, color="#00b894")
    ax.plot(hours, rl_result["energy"], "s-", color="#00b894",
            label="RL Agent",  linewidth=2, markersize=5)

    # Shade night‑time hours
    ax.axvspan(0, 6, alpha=0.06, color="navy", label="Night hours")
    ax.axvspan(21, 24, alpha=0.06, color="navy")

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Energy per hour (Wh)")
    ax.set_title("Hourly Energy Consumption Profile")
    ax.set_xticks(hours)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig("results/3_hourly_energy.png", dpi=150)
    plt.close(fig)
    print("  -> Saved results/3_hourly_energy.png")


def plot_cell_decisions(rl_result):
    """Plot 4 — Heatmap of small-cell ON/OFF decisions over 24 hours."""
    status = rl_result["status"]   # shape: (24, num_cells)
    traffic = rl_result["traffic"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6),
                                    gridspec_kw={"height_ratios": [1, 1.4]},
                                    sharex=True)

    # Top: traffic per cell
    hours = np.arange(cfg.STEPS_PER_EPISODE)
    for i in range(cfg.NUM_SMALL_CELLS):
        ax1.plot(hours, traffic[:, i], marker=".", label=f"Cell {i+1}")
    ax1.set_ylabel("Traffic (Mbps)")
    ax1.set_title("Traffic Load & RL Agent's ON/OFF Decisions")
    ax1.legend(loc="upper right", fontsize=8, ncol=cfg.NUM_SMALL_CELLS)

    # Bottom: ON/OFF heatmap
    cmap = plt.cm.colors.ListedColormap(["#d63031", "#00b894"])
    im = ax2.imshow(status.T, aspect="auto", cmap=cmap,
                     interpolation="nearest", vmin=0, vmax=1)
    ax2.set_yticks(range(cfg.NUM_SMALL_CELLS))
    ax2.set_yticklabels([f"Cell {i+1}" for i in range(cfg.NUM_SMALL_CELLS)])
    ax2.set_xlabel("Hour of Day")
    ax2.set_xticks(hours)

    # Colour‑bar legend
    cbar = fig.colorbar(im, ax=ax2, ticks=[0.25, 0.75], shrink=0.6)
    cbar.ax.set_yticklabels(["OFF", "ON"])

    fig.tight_layout()
    fig.savefig("results/4_cell_decisions.png", dpi=150)
    plt.close(fig)
    print("  -> Saved results/4_cell_decisions.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4.  MAIN ENTRY POINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    # ---------- Phase 1: Train ----------
    agent, rewards, energies = train()

    # ---------- Phase 2: Evaluate ----------
    print("=" * 60)
    print("  EVALUATION - Comparing policies over a 24-hour day")
    print("=" * 60)

    rl_result   = evaluate(agent, policy="rl")
    on_result   = evaluate(agent, policy="always_on")
    rand_result = evaluate(agent, policy="random")

    # ---------- Save report-ready tables ----------
    save_metrics_csv(rl_result, "rl", "results/metrics_rl.csv")
    save_metrics_csv(on_result, "always_on", "results/metrics_always_on.csv")
    save_metrics_csv(rand_result, "random", "results/metrics_random.csv")
    save_summary_txt(rl_result, on_result, rand_result, "results/summary.txt")
    print("  -> Saved results/metrics_*.csv and results/summary.txt")

    rl_total   = rl_result["energy"].sum()
    on_total   = on_result["energy"].sum()
    rand_total = rand_result["energy"].sum()

    print(f"\n  Always ON  : {on_total:>8.1f} Wh")
    print(f"  Random     : {rand_total:>8.1f} Wh")
    print(f"  RL Agent   : {rl_total:>8.1f} Wh")
    print(f"  -------------------------------")
    print(f"  Energy Saved (vs Always ON): "
          f"{on_total - rl_total:.1f} Wh  "
          f"({(on_total - rl_total) / on_total * 100:.1f}%)")
    print(f"  Dropped traffic (RL): "
          f"{rl_result['dropped'].sum():.2f} Mbps total\n")

    # ---------- Phase 3: Plots ----------
    print("  Generating plots ...")
    plot_training_curve(rewards)
    plot_energy_comparison(rl_result, on_result, rand_result)
    plot_hourly_energy(rl_result, on_result)
    plot_cell_decisions(rl_result)

    print("\n" + "=" * 60)
    print("  ALL DONE!  Check the results/ folder for plots.")
    print("=" * 60)
