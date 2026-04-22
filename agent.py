import numpy as np
import os
import config as cfg

class QLearningAgent:
    def __init__(self, num_cells, num_actions):
        self.num_cells = num_cells
        self.num_actions = num_actions
        self.lr = cfg.LEARNING_RATE
        self.gamma = cfg.DISCOUNT_FACTOR
        self.epsilon = cfg.EPSILON_START
        self.epsilon_min = cfg.EPSILON_END
        self.epsilon_decay = cfg.EPSILON_DECAY
        self.traffic_bins = cfg.TRAFFIC_BINS
        self.time_bins = cfg.TIME_BINS
        dims = tuple([self.traffic_bins] * self.num_cells + [self.time_bins])
        self.q_table = np.zeros(dims + (self.num_actions,))

    def discretize(self, state):
        traffic = state[: self.num_cells]
        norm_h = state[-1]
        mx_traffic = cfg.TRAFFIC_BASE_LOAD + cfg.TRAFFIC_AMPLITUDE + cfg.TRAFFIC_NOISE_STD * 3
        t_indices = np.clip((traffic / mx_traffic * self.traffic_bins).astype(int), 0, self.traffic_bins - 1)
        h_index = min(int(norm_h * self.time_bins), self.time_bins - 1)
        return tuple(t_indices) + (h_index,)

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        idx = self.discretize(state)
        return int(np.argmax(self.q_table[idx]))

    def learn(self, state, action, reward, next_state, done):
        s = self.discretize(state)
        s_next = self.discretize(next_state)
        old_q = self.q_table[s + (action,)]
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[s_next])
        self.q_table[s + (action,)] = old_q + self.lr * (target - old_q)

    def decay(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def select_best(self, state):
        idx = self.discretize(state)
        return int(np.argmax(self.q_table[idx]))

    def save(self, path="results/q_table.npy"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self.q_table)

    def load(self, path="results/q_table.npy"):
        self.q_table = np.load(path)
        self.epsilon = self.epsilon_min
