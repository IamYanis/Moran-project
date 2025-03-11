import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

file_path = "data/absorption_times.csv"
if not os.path.exists(file_path):
    print(f"Error: '{file_path}' not found.")
    exit()

abs_data = pd.read_csv(file_path)

if "N" not in abs_data.columns or "r" not in abs_data.columns or "AbsorptionTime" not in abs_data.columns:
    print("Error: Missing required columns in absorption_times.csv")
    exit()

selected_N = [50, 100, 300, 1000]
r_values = sorted(abs_data["r"].unique())

fig, axes = plt.subplots(1, len(selected_N), figsize=(15, 5), sharey=True)

L_range = 150 

for i, N in enumerate(selected_N):
    subset = abs_data[abs_data["N"] == N]
    for r in r_values:
        times = subset[subset["r"] == r]["AbsorptionTime"]
        if times.empty:
            continue
        axes[i].hist(times, bins=L_range, alpha=0.5, histtype='stepfilled', label=f"r={r:.2f}", edgecolor="black", range=(0, L_range))

    axes[i].set_xlabel("Absorption Time")
    axes[i].set_title(f"N = {N}")
    axes[i].legend()

axes[0].set_ylabel("Frequency")
plt.suptitle("Histograms of Absorption Times")
plt.tight_layout()
plt.show()
