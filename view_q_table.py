import numpy as np

# Load the Q-table
q_table = np.load("results/q_table.npy")

print("=" * 40)
print("       Q-TABLE INFORMATION")
print("=" * 40)
print(f"Shape: {q_table.shape}")
print(f"Total entries: {q_table.size:,}")
print(f"Data type: {q_table.dtype}")

# Basic statistics
print("\n[Statistics]")
print(f"Min value: {np.min(q_table):.4f}")
print(f"Max value: {np.max(q_table):.4f}")
print(f"Mean value: {np.mean(q_table):.4f}")
print(f"Non-zero entries: {np.count_nonzero(q_table):,}")

# Show a sample state
# For example: [Low Traffic, Low, Low, Low, Low] during [Night]
# State index: (0, 0, 0, 0, 0, 0)
sample_idx = (0, 0, 0, 0, 0, 0)
q_values = q_table[sample_idx]
best_action = np.argmax(q_values)

print(f"\n[Sample State: {sample_idx}]")
print("Q-values for all 32 actions:")
print(q_values)
print(f"\nBest Action for this state: {best_action}")
print(f"Binary of action {best_action}: {bin(best_action)[2:].zfill(5)} (1=ON, 0=OFF)")

print("\n" + "=" * 40)
