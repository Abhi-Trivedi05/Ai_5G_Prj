import numpy as np
import config as cfg

class SmallCellNetwork:
    def __init__(self):
        self.num_cells = cfg.NUM_SMALL_CELLS
        self.num_actions = 2 ** self.num_cells
        self.state_size = 2 * self.num_cells + 1
        self.current_step = 0
        self.cell_status = np.ones(self.num_cells, dtype=int)
        self.traffic = np.zeros(self.num_cells)

    def reset(self):
        self.current_step = 0
        self.cell_status = np.ones(self.num_cells, dtype=int)
        self.traffic = self._generate_traffic(0)
        return self._get_state()

    def step(self, action):
        self.cell_status = self._decode_action(action)
        energy, cell_energies = self._compute_energy()
        overflow = self._compute_overflow()
        dropped = max(0.0, overflow - cfg.MACRO_CELL_CAPACITY)
        reward = -(cfg.ENERGY_REWARD_SCALE * energy + cfg.QOS_PENALTY_WEIGHT * dropped)
        self.current_step += 1
        done = self.current_step >= cfg.STEPS_PER_EPISODE
        if not done:
            self.traffic = self._generate_traffic(self.current_step)
        next_state = self._get_state()
        info = {
            "energy": energy,
            "cell_energies": cell_energies,
            "dropped_traffic": dropped,
            "overflow_traffic": overflow,
            "traffic": self.traffic.copy(),
            "cell_status": self.cell_status.copy(),
        }
        return next_state, reward, done, info

    def _generate_traffic(self, hour):
        phase = 2.0 * np.pi * (hour - cfg.TRAFFIC_PEAK_HOUR) / 24.0
        base = cfg.TRAFFIC_BASE_LOAD + cfg.TRAFFIC_AMPLITUDE * (0.5 * (1.0 + np.cos(phase)))
        noise = np.random.normal(0, cfg.TRAFFIC_NOISE_STD, self.num_cells)
        traffic = np.full(self.num_cells, base) + noise
        return np.clip(traffic, 0.0, None)

    def _compute_energy(self):
        max_rate = cfg.TRAFFIC_BASE_LOAD + cfg.TRAFFIC_AMPLITUDE
        cell_pours = np.zeros(self.num_cells)
        for i in range(self.num_cells):
            if self.cell_status[i] == 1:
                ratio = min(self.traffic[i] / max_rate, 1.0)
                cell_pours[i] = cfg.SMALL_CELL_ACTIVE_POWER + cfg.SMALL_CELL_LOAD_COEFF * ratio
            else:
                cell_pours[i] = cfg.SMALL_CELL_SLEEP_POWER
        overflow = self._compute_overflow()
        m_ratio = min(overflow / cfg.MACRO_CELL_CAPACITY, 1.0)
        macro_power = cfg.MACRO_CELL_BASE_POWER + cfg.MACRO_CELL_LOAD_COEFF * m_ratio
        return cell_pours.sum() + macro_power, cell_pours

    def _compute_overflow(self):
        overflow = 0.0
        for i in range(self.num_cells):
            if self.cell_status[i] == 0:
                overflow += self.traffic[i]
        return overflow

    def _get_state(self):
        norm_h = self.current_step / cfg.STEPS_PER_EPISODE
        return np.concatenate([self.traffic, self.cell_status.astype(float), [norm_h]])

    def _decode_action(self, action):
        bits = []
        for _ in range(self.num_cells):
            bits.append(action % 2)
            action //= 2
        return np.array(bits, dtype=int)
