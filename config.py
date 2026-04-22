"""
config.py — Central Configuration for 5G RAN Energy Optimization
=================================================================
All simulation parameters are defined here so they can be easily
tweaked without touching any other file.
"""

# ─────────────────────────────────────────────
# 1. NETWORK TOPOLOGY
# ─────────────────────────────────────────────

NUM_SMALL_CELLS = 5          # Number of small cells the RL agent controls
MACRO_CELL_CAPACITY = 100.0  # Max traffic the macro cell can handle (Mbps)

# Reproducibility
SEED = 42


# ─────────────────────────────────────────────
# 2. POWER MODEL  (Watts)
# ─────────────────────────────────────────────
# Each small cell consumes power when active and a fraction when sleeping.
# The macro cell is always on and consumes more power.

SMALL_CELL_ACTIVE_POWER  = 6.0    # Base power when a small cell is ON  (W)
SMALL_CELL_LOAD_COEFF    = 4.0    # Extra power per unit traffic load   (W)
SMALL_CELL_SLEEP_POWER   = 0.5    # Standby power when a cell is OFF    (W)

MACRO_CELL_BASE_POWER    = 40.0   # Baseline power of the macro cell    (W)
MACRO_CELL_LOAD_COEFF    = 8.0    # Extra power per unit traffic load   (W)


# ─────────────────────────────────────────────
# 3. TRAFFIC MODEL
# ─────────────────────────────────────────────
# Traffic follows a sinusoidal day/night pattern.
# peak_hour   = hour of max traffic (e.g., 14 = 2 PM)
# base_load   = minimum traffic even at night  (Mbps)
# amplitude   = swing between min and max traffic (Mbps)
# noise_std   = random variation per cell per step (Mbps)

TRAFFIC_PEAK_HOUR  = 14.0    # Hour of the day with maximum traffic
TRAFFIC_BASE_LOAD  = 5.0     # Minimum traffic per small cell (Mbps)
TRAFFIC_AMPLITUDE  = 15.0    # Half-range of the sinusoidal variation
TRAFFIC_NOISE_STD  = 2.0     # Gaussian noise added to each cell's traffic

# A conservative upper bound used for normalization/binning.
# (Mean + amplitude + ~3σ noise) keeps ratios/bins stable under noise.
TRAFFIC_MAX_EST = TRAFFIC_BASE_LOAD + TRAFFIC_AMPLITUDE + 3.0 * TRAFFIC_NOISE_STD


# ─────────────────────────────────────────────
# 4. QUALITY OF SERVICE (QoS)
# ─────────────────────────────────────────────
# If traffic is offloaded to the macro cell and it exceeds capacity,
# users experience dropped connections → big penalty.

QOS_PENALTY_WEIGHT = 50.0    # Penalty multiplier for each Mbps of dropped traffic
ENERGY_REWARD_SCALE = 0.1    # Scale factor so energy and QoS penalty are comparable


# ─────────────────────────────────────────────
# 5. SIMULATION
# ─────────────────────────────────────────────

STEPS_PER_EPISODE = 24       # Each step = 1 hour → one episode = 24 hours
NUM_EPISODES      = 800      # Total training episodes


# ─────────────────────────────────────────────
# 6. Q-LEARNING AGENT
# ─────────────────────────────────────────────

LEARNING_RATE     = 0.1      # α  — how fast the agent updates Q-values
DISCOUNT_FACTOR   = 0.95     # γ  — importance of future rewards
EPSILON_START     = 1.0      # Initial exploration probability
EPSILON_END       = 0.05     # Minimum exploration probability
EPSILON_DECAY     = 0.997    # Multiply epsilon by this after each episode

# Discretization bins for the state space
TRAFFIC_BINS = 3             # low / medium / high traffic per cell
TIME_BINS    = 6             # Split 24 hours into 6 buckets (4-hour blocks)
