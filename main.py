import os
import numpy as np
import matplotlib.pyplot as plt
import config as cfg
from environment import SmallCellNetwork
from agent import QLearningAgent

os.makedirs("results", exist_ok=True)

def train():
    env = SmallCellNetwork()
    agent = QLearningAgent(env.num_cells, env.num_actions)
    rewards = []
    energies = []

    print("=" * 60)
    print("  TRAINING STARTED")
    print("=" * 60)

    for ep in range(1, cfg.NUM_EPISODES + 1):
        state = env.reset()
        t_reward = 0
        t_energy = 0
        for _ in range(cfg.STEPS_PER_EPISODE):
            action = agent.choose_action(state)
            next_state, reward, _, info = env.step(action)
            agent.learn(state, action, reward, next_state, _ == cfg.STEPS_PER_EPISODE - 1)
            t_reward += reward
            t_energy += info["energy"]
            state = next_state
        agent.decay()
        rewards.append(t_reward)
        energies.append(t_energy)
        if ep % 100 == 0:
            print(f"  Episode {ep} | Reward: {np.mean(rewards[-100:]):.1f} | Energy: {np.mean(energies[-100:]):.1f} Wh")

    agent.save()
    return agent, rewards, energies

def evaluate(agent, policy="rl"):
    env = SmallCellNetwork()
    state = env.reset()
    res = {"energy": [], "traffic": [], "status": [], "dropped": []}
    for _ in range(cfg.STEPS_PER_EPISODE):
        if policy == "rl": action = agent.select_best(state)
        elif policy == "always_on": action = env.num_actions - 1
        else: action = np.random.randint(env.num_actions)
        next_state, _, _, info = env.step(action)
        res["energy"].append(info["energy"])
        res["traffic"].append(info["traffic"])
        res["status"].append(info["cell_status"])
        res["dropped"].append(info["dropped_traffic"])
        state = next_state
    for k in res: res[k] = np.array(res[k])
    return res

def plot_all(rewards, rl, on, rand):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.3, color="blue")
    plt.title("Reward Curve")
    plt.savefig("results/1_training_curve.png")
    
    plt.figure(figsize=(8, 5))
    plt.bar(["Always ON", "Random", "RL"], [on["energy"].sum(), rand["energy"].sum(), rl["energy"].sum()])
    plt.title("Energy Comparison")
    plt.savefig("results/2_energy_comparison.png")
    
    plt.figure(figsize=(10, 5))
    plt.plot(on["energy"], label="Always ON")
    plt.plot(rl["energy"], label="RL Agent")
    plt.legend()
    plt.savefig("results/3_hourly_energy.png")
    
    plt.figure(figsize=(12, 6))
    plt.imshow(rl["status"].T, aspect="auto")
    plt.savefig("results/4_cell_decisions.png")
    plt.close("all")

if __name__ == "__main__":
    agent, rewards, _ = train()
    rl = evaluate(agent, "rl")
    on = evaluate(agent, "always_on")
    rand = evaluate(agent, "random")
    print(f"Always ON: {on['energy'].sum():.1f} Wh")
    print(f"RL Agent: {rl['energy'].sum():.1f} Wh")
    plot_all(rewards, rl, on, rand)
    print("Done. Results in results folder.")
