# Energy-Efficient 5G RAN Optimization using Reinforcement Learning

A student-friendly implementation demonstrating how **Reinforcement Learning (Q-Learning)** can optimize energy consumption in a **5G Radio Access Network (RAN)** by intelligently switching small cells ON/OFF based on traffic demand.

---

## 📌 Project Overview

### The Problem
5G networks deploy many **small cells** alongside a **macro cell** to provide high-speed coverage. However, keeping all small cells active 24/7 wastes enormous energy — especially during low-traffic hours (e.g., midnight to early morning). This project addresses the question:

> *Can an AI agent learn when to turn OFF unnecessary small cells to save energy without degrading service quality?*

### The Solution
We use **Q-Learning** (a Reinforcement Learning technique) to train an agent that:
- Observes the **current traffic load** on each small cell
- Decides which cells to **turn ON or OFF**
- Minimizes **total energy consumption** while ensuring the macro cell can handle overflow traffic

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────┐
│                    main.py                           │
│          (Training Loop + Evaluation + Plots)        │
├──────────────────┬───────────────────────────────────┤
│                  │                                   │
│   ┌──────────────▼──────────────┐  ┌────────────────▼─────────────┐
│   │         agent.py            │  │       environment.py         │
│   │    Q-Learning Agent         │  │    5G Network Simulator      │
│   │                             │  │                              │
│   │  • Q-table (state→action)   │  │  • 5 Small Cells (ON/OFF)   │
│   │  • ε-greedy exploration     │  │  • 1 Macro Cell (always ON)  │
│   │  • Learn from rewards       │  │  • Sinusoidal traffic model  │
│   │  • Decay exploration        │  │  • Power consumption model   │
│   └──────────────▲──────────────┘  │  • QoS penalty for drops     │
│                  │                 └────────────────▲─────────────┘
│                  │                                  │
│              ┌───┴──────────────────────────────────┘
│              │         config.py
│              │   (All parameters in one place)
│              └────────────────────────────────
└──────────────────────────────────────────────────────┘
```

---

## 📂 File Structure

| File | Purpose |
|------|---------|
| `config.py` | All tunable parameters (network, traffic, energy, RL agent) |
| `environment.py` | 5G RAN simulator with Gym-style `reset()`/`step()` API |
| `agent.py` | Q-Learning agent with ε-greedy exploration |
| `main.py` | Training loop, baseline comparison, and plot generation |
| `requirements.txt` | Python dependencies |
| `README.md` | This documentation file |

---

## 🚀 How to Run

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the project
```bash
python main.py
```

### What happens when you run it:
1. **Training Phase** — The Q-learning agent trains for 800 episodes (takes ~30 seconds)
2. **Evaluation Phase** — Compares the trained agent against "Always ON" and "Random" baselines
3. **Plot Generation** — Creates 4 graphs in the `results/` folder

### Output Structure
```
results/
├── q_table.npy                # Saved Q-table (trained model)
├── 1_training_curve.png       # Reward convergence over episodes
├── 2_energy_comparison.png    # Bar chart: RL vs Always-ON vs Random
├── 3_hourly_energy.png        # 24-hour energy profile
└── 4_cell_decisions.png       # Heatmap of ON/OFF decisions + traffic
```

---

## 🔬 How the System Works (Step-by-Step)

### 1. Traffic Generation
Traffic follows a **sinusoidal day/night pattern**:
- **Peak traffic** at 2:00 PM (configurable)
- **Low traffic** at 2:00 AM
- Each cell gets slightly different traffic via random noise

### 2. Energy Model
Each small cell consumes:
- **When ON**: `6W + 4W × (traffic_load / max_traffic)`
- **When OFF**: `0.5W` (standby)

The macro cell is always on: `40W + 8W × (overflow_ratio)`

### 3. State Space
The agent observes:
- Traffic load on each of the 5 small cells
- Current ON/OFF status of each cell
- Normalized time of day (0.0 to 1.0)

### 4. Action Space
The agent outputs an integer (0–31) whose **binary representation** maps to ON/OFF:
```
Action 31 = 11111 → All 5 cells ON
Action  0 = 00000 → All 5 cells OFF
Action 19 = 10011 → Cells 0,1,4 ON; Cells 2,3 OFF
```

### 5. Reward Function
```
reward = -(0.1 × total_energy + 50 × dropped_traffic)
```
- Negative because we want to **minimize** energy
- Large penalty if macro cell can't absorb overflow → prevents QoS violations

### 6. Q-Learning Update
```
Q(s, a) ← Q(s, a) + α [r + γ · max Q(s', a') − Q(s, a)]
```
Where α=0.1 (learning rate), γ=0.95 (discount factor)

---

## 📊 Expected Results

After training, the RL agent typically achieves:
- **15-30% energy savings** compared to the "Always ON" baseline
- **Near-zero dropped traffic** (QoS maintained)
- **Intelligent behaviour**: turns OFF cells during night hours, turns them back ON during peak hours

---

## ⚙️ Configuration

All parameters can be tweaked in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_SMALL_CELLS` | 5 | Number of controllable small cells |
| `NUM_EPISODES` | 800 | Training episodes |
| `LEARNING_RATE` | 0.1 | Q-learning step size (α) |
| `DISCOUNT_FACTOR` | 0.95 | Future reward importance (γ) |
| `EPSILON_START` | 1.0 | Initial exploration rate |
| `QOS_PENALTY_WEIGHT` | 50.0 | Penalty for dropped traffic |

---

## 🎓 Viva Preparation Notes

### Key Concepts to Explain
1. **Why RL?** — The problem is sequential (decisions at each hour affect future state) and the optimal policy is unknown → perfect fit for RL
2. **Why Q-Learning?** — Simple, well-understood, no neural network needed for this state space size
3. **State discretization** — We bin continuous traffic into low/medium/high to make the Q-table finite
4. **ε-greedy** — Balance between exploring new actions and exploiting known good ones
5. **Reward shaping** — Energy cost + QoS penalty ensures the agent saves energy without dropping users

### Likely Viva Questions
- *"Why not use Deep Q-Network?"* → Our state space is small enough for tabular Q-learning. DQN would be overkill and harder to interpret.
- *"How do you ensure QoS?"* → The reward function heavily penalizes dropped traffic, so the agent learns to avoid turning off too many cells.
- *"What if traffic patterns change?"* → Retrain the agent on new data. The sinusoidal model can be adjusted via config.py.
- *"How is this different from a rule-based approach?"* → The RL agent discovers the optimal policy automatically without manual threshold tuning.

---

## 📄 License
This project is created for educational purposes as a final-year project demonstration.
