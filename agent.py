"""
agent.py — Q-Learning Agent for 5G Energy Optimization
========================================================
A tabular Q‑learning agent that learns which small cells to
turn ON or OFF at each hour of the day to minimize total
network energy consumption.

The continuous state space is discretized into bins so that
a simple lookup table (Q‑table) can be used instead of a
neural network.
"""

import numpy as np
import os
import config as cfg


class QLearningAgent:
    """
    Tabular Q‑Learning agent.

    State discretization
    --------------------
    • Traffic per cell  → TRAFFIC_BINS levels  (low / medium / high)
    • Hour of day       → TIME_BINS buckets

    Action space
    ------------
    • 2^N actions, one for each ON/OFF combination of N small cells.
    """

    def __init__(self, num_cells, num_actions, seed=None):
        self.num_cells = num_cells
        self.num_actions = num_actions
        self.rng = np.random.default_rng(cfg.SEED if seed is None else seed)

        # Hyper‑parameters (from config)
        self.lr = cfg.LEARNING_RATE
        self.gamma = cfg.DISCOUNT_FACTOR
        self.epsilon = cfg.EPSILON_START
        self.epsilon_min = cfg.EPSILON_END
        self.epsilon_decay = cfg.EPSILON_DECAY

        # Discretization settings
        self.traffic_bins = cfg.TRAFFIC_BINS
        self.time_bins = cfg.TIME_BINS

        # Build the Q‑table
        # State is encoded as:
        #   (traffic_bin_cell0..N, status_cell0..N, time_bin)
        # Total discrete states = (TRAFFIC_BINS^N) × (2^N) × TIME_BINS
        dims = tuple([self.traffic_bins] * self.num_cells + [2] * self.num_cells + [self.time_bins])
        self.q_table = np.zeros(dims + (self.num_actions,))
        print(f"[Agent] Q-table shape: {self.q_table.shape}  "
              f"({self.q_table.size:,} entries)")

    # ──────────────────────────────────────
    # Discretize continuous state → table index
    # ──────────────────────────────────────
    def discretize_state(self, state):
        """
        Convert a continuous state vector into a tuple of bin indices
        that can be used to index the Q‑table.

        Parameters
        ----------
        state : np.array
            [traffic_0, …, traffic_N, status_0, …, status_N, norm_hour]

        Returns
        -------
        tuple of ints — one index per dimension of the Q‑table.
        """
        traffic = state[: self.num_cells]
        status = state[self.num_cells: 2 * self.num_cells]
        norm_hour = state[-1]  # already in [0, 1]

        # Traffic bins: divide the expected traffic range into equal parts
        max_traffic = max(float(cfg.TRAFFIC_MAX_EST), 1e-9)
        traffic_indices = np.clip(
            (traffic / max_traffic * self.traffic_bins).astype(int),
            0, self.traffic_bins - 1,
        )

        status_indices = np.clip(status.astype(int), 0, 1)

        # Time bin
        time_index = min(
            int(norm_hour * self.time_bins),
            self.time_bins - 1,
        )

        return tuple(traffic_indices) + tuple(status_indices) + (time_index,)

    # ──────────────────────────────────────
    # Action selection (ε‑greedy)
    # ──────────────────────────────────────
    def choose_action(self, state):
        """
        Pick an action using epsilon‑greedy strategy:
          • With probability ε  → random action  (explore)
          • Otherwise           → best Q‑value   (exploit)
        """
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.num_actions))

        state_idx = self.discretize_state(state)
        q_values = self.q_table[state_idx]
        return int(np.argmax(q_values))

    # ──────────────────────────────────────
    # Q‑value update
    # ──────────────────────────────────────
    def learn(self, state, action, reward, next_state, done):
        """
        Standard Q‑learning update:
            Q(s, a) ← Q(s, a) + α [ r + γ·max_a' Q(s', a') − Q(s, a) ]
        """
        s = self.discretize_state(state)
        s_next = self.discretize_state(next_state)

        current_q = self.q_table[s + (action,)]

        if done:
            target = reward     # no future reward at terminal state
        else:
            target = reward + self.gamma * np.max(self.q_table[s_next])

        # Update rule
        self.q_table[s + (action,)] = current_q + self.lr * (target - current_q)

    # ──────────────────────────────────────
    # Decay exploration rate
    # ──────────────────────────────────────
    def decay_epsilon(self):
        """Reduce exploration rate after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ──────────────────────────────────────
    # Greedy action (no exploration) for evaluation
    # ──────────────────────────────────────
    def best_action(self, state):
        """Pick the action with the highest Q‑value (no randomness)."""
        state_idx = self.discretize_state(state)
        return int(np.argmax(self.q_table[state_idx]))

    # ──────────────────────────────────────
    # Save / Load
    # ──────────────────────────────────────
    def save(self, path="results/q_table.npy"):
        """Save the trained Q‑table to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self.q_table)
        print(f"[Agent] Q-table saved to {path}")

    def load(self, path="results/q_table.npy"):
        """Load a previously trained Q‑table."""
        self.q_table = np.load(path)
        self.epsilon = self.epsilon_min   # no exploration after loading
        print(f"[Agent] Q-table loaded from {path}")
