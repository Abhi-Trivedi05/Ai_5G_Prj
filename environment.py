"""
environment.py — 5G RAN Network Simulation Environment
========================================================
Simulates a small‑cell network where an RL agent decides which
small cells to turn ON or OFF every hour to save energy, while
a macro cell absorbs any overflow traffic.

Follows the OpenAI‑Gym style:  reset() → step(action) → (state, reward, done, info)
"""

import numpy as np
import config as cfg


class SmallCellNetwork:
    """
    A simplified 5G Radio Access Network with:
      • N small cells (can be switched ON/OFF by the agent)
      • 1 macro cell  (always ON, absorbs overflow traffic)
    """

    def __init__(self, seed=None):
        self.num_cells = cfg.NUM_SMALL_CELLS

        # The action space size: each small cell can be ON(1) or OFF(0)
        # so there are 2^N possible actions  (e.g. 2^5 = 32)
        self.num_actions = 2 ** self.num_cells

        # State vector length:
        #   [traffic_per_cell (N), cell_on_off (N), time_of_day_normalized (1)]
        self.state_size = 2 * self.num_cells + 1

        # Internal clock
        self.current_step = 0
        self.cell_status = np.ones(self.num_cells, dtype=int)  # all ON initially
        self.traffic = np.zeros(self.num_cells)
        self.rng = np.random.default_rng(cfg.SEED if seed is None else seed)

    # ──────────────────────────────────────
    # Reset — start a new 24‑hour episode
    # ──────────────────────────────────────
    def reset(self, seed=None):
        """Reset the environment for a new episode (a new day)."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.current_step = 0
        self.cell_status = np.ones(self.num_cells, dtype=int)
        self.traffic = self._generate_traffic(hour=0)
        return self._get_state()

    # ──────────────────────────────────────
    # Step — advance one hour
    # ──────────────────────────────────────
    def step(self, action):
        """
        Execute one time‑step (1 hour).

        Parameters
        ----------
        action : int
            An integer 0 … 2^N−1 whose binary representation
            tells which small cells are ON (1) or OFF (0).
            Example for 5 cells:  action=19 → binary 10011
              → cells 0,1 ON; cell 2 OFF; cell 3 OFF; cell 4 ON

        Returns
        -------
        next_state : np.array   — the new observation
        reward     : float      — negative of (energy + QoS penalty)
        done       : bool       — True when 24 steps are over
        info       : dict       — extra diagnostics
        """

        # 1. Decode the action into ON/OFF decisions
        self.cell_status = self._decode_action(action)

        # 2. Calculate energy consumption
        energy, cell_energies = self._compute_energy()

        # 3. Check QoS — can the macro cell handle the overflow?
        overflow_traffic = self._compute_overflow()
        dropped_traffic = max(0.0, overflow_traffic - cfg.MACRO_CELL_CAPACITY)

        # 4. Compute reward  (we want to minimise energy while avoiding drops)
        qos_penalty = cfg.QOS_PENALTY_WEIGHT * dropped_traffic
        reward = -(cfg.ENERGY_REWARD_SCALE * energy + qos_penalty)

        # 5. Advance the clock
        self.current_step += 1
        done = self.current_step >= cfg.STEPS_PER_EPISODE

        # 6. Generate traffic for the next hour
        if not done:
            self.traffic = self._generate_traffic(hour=self.current_step)

        next_state = self._get_state()

        info = {
            "energy":           energy,
            "cell_energies":    cell_energies,
            "dropped_traffic":  dropped_traffic,
            "overflow_traffic": overflow_traffic,
            "traffic":          self.traffic.copy(),
            "cell_status":      self.cell_status.copy(),
        }

        return next_state, reward, done, info

    # ──────────────────────────────────────
    # Traffic generation (sinusoidal + noise)
    # ──────────────────────────────────────
    def _generate_traffic(self, hour):
        """
        Create a realistic traffic load for each small cell.

        The pattern follows a sine wave peaking at TRAFFIC_PEAK_HOUR,
        with random noise to make each cell slightly different.
        """
        # Sinusoidal component: ranges from (base - amp) to (base + amp)
        phase = 2.0 * np.pi * (hour - cfg.TRAFFIC_PEAK_HOUR) / 24.0
        base_traffic = cfg.TRAFFIC_BASE_LOAD + cfg.TRAFFIC_AMPLITUDE * (
            0.5 * (1.0 + np.cos(phase))
        )

        # Add per‑cell random noise
        noise = self.rng.normal(0, cfg.TRAFFIC_NOISE_STD, self.num_cells)
        traffic = np.full(self.num_cells, base_traffic) + noise

        # Traffic can't be negative
        return np.clip(traffic, 0.0, None)

    # ──────────────────────────────────────
    # Power consumption model
    # ──────────────────────────────────────
    def _compute_energy(self):
        """
        Calculate total network energy for this hour.

        For each small cell:
            • If ON:  power = base_power + load_coeff × (traffic / max_traffic)
            • If OFF: power = sleep_power  (small standby cost)

        The macro cell is always on, plus extra cost for absorbed traffic.
        """
        max_traffic = max(float(cfg.TRAFFIC_MAX_EST), 1e-9)
        cell_energies = np.zeros(self.num_cells)

        for i in range(self.num_cells):
            if self.cell_status[i] == 1:  # cell is ON
                load_ratio = min(self.traffic[i] / max_traffic, 1.0)
                cell_energies[i] = (
                    cfg.SMALL_CELL_ACTIVE_POWER
                    + cfg.SMALL_CELL_LOAD_COEFF * load_ratio
                )
            else:  # cell is OFF (sleep mode)
                cell_energies[i] = cfg.SMALL_CELL_SLEEP_POWER

        # Macro cell energy  (always ON; extra load from offloaded traffic)
        overflow = self._compute_overflow()
        macro_load_ratio = min(overflow / cfg.MACRO_CELL_CAPACITY, 1.0)
        macro_energy = (
            cfg.MACRO_CELL_BASE_POWER
            + cfg.MACRO_CELL_LOAD_COEFF * macro_load_ratio
        )

        total_energy = cell_energies.sum() + macro_energy
        return total_energy, cell_energies

    # ──────────────────────────────────────
    # Overflow — traffic that sleeping cells push to the macro cell
    # ──────────────────────────────────────
    def _compute_overflow(self):
        """
        When a small cell is OFF, its traffic must be handled by
        the macro cell.  Sum up all such offloaded traffic.
        """
        overflow = 0.0
        for i in range(self.num_cells):
            if self.cell_status[i] == 0:
                overflow += self.traffic[i]
        return overflow

    # ──────────────────────────────────────
    # State observation
    # ──────────────────────────────────────
    def _get_state(self):
        """
        Build the state vector the agent observes:
          [ traffic_cell_0, …, traffic_cell_N,
            status_cell_0,  …, status_cell_N,
            normalised_hour ]
        """
        norm_hour = self.current_step / cfg.STEPS_PER_EPISODE
        state = np.concatenate([
            self.traffic,
            self.cell_status.astype(float),
            [norm_hour],
        ])
        return state

    # ──────────────────────────────────────
    # Decode integer action → binary ON/OFF
    # ──────────────────────────────────────
    def _decode_action(self, action):
        """
        Convert an integer action to a binary array.
        Example:  action=19, 5 cells →  [1, 1, 0, 0, 1]
        """
        return np.array([(action >> i) & 1 for i in range(self.num_cells)], dtype=int)
