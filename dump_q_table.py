import numpy as np
import os

# Load the binary Q-table
q_table = np.load("results/q_table.npy")

# We will save it as a text file where each state is clearly labeled
with open("results/q_table_full.txt", "w") as f:
    f.write("FULL Q-TABLE DATA DUMP\n")
    f.write("Format: [Cell1, Cell2, Cell3, Cell4, Cell5, Time] -> [32 Action Values]\n")
    f.write("-" * 60 + "\n\n")

    # Iterate through all states
    it = np.nditer(q_table[..., 0], flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        q_values = q_table[idx]
        
        # Only write states that have been modified (learned) to keep the file size reasonable
        if np.any(q_values != 0):
            line = f"State {idx}: " + str(q_values.tolist()) + "\n"
            f.write(line)
            
        it.iternext()

print("Success! Created results/q_table_full.txt")
print("You can now open this file in any text editor to see the raw numbers.")
