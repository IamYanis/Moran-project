import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Load only necessary columns
cols_to_read = ["N", "r", "Type", "Value", "std_sup_dev"]

# Read CSV in chunks to avoid memory overload
chunks = pd.read_csv("data/moran_simulation_results.csv", usecols=cols_to_read, chunksize=100000)
filtered_chunks = [chunk[chunk["Type"] == "Deviation"] for chunk in chunks]
dev_data = pd.concat(filtered_chunks, ignore_index=True)

# Convert data types to save memory
dev_data["N"] = pd.to_numeric(dev_data["N"], downcast="integer")
dev_data["r"] = pd.to_numeric(dev_data["r"], downcast="float")
dev_data["Value"] = pd.to_numeric(dev_data["Value"], downcast="float")
dev_data["std_sup_dev"] = pd.to_numeric(dev_data["std_sup_dev"], downcast="float")

# Extract values
N_values = sorted(dev_data["N"].unique())
r_values = sorted(dev_data["r"].unique())

# Reduce number of plotted points
if len(N_values) > 50:
    dev_data = dev_data.sample(n=50, random_state=42)

n_sim = 1000  # This should match N_SIM in main.c

# Deviation plots
fig1, axes = plt.subplots(1, 2, figsize=(12, 5))
scaling_exponents = {}

for r in r_values:
    subset = dev_data[dev_data["r"] == r]

    N = subset["N"]
    mean_dev = subset["Value"]
    variance_dev = subset["std_sup_dev"] ** 2

    std_err = np.sqrt(variance_dev / n_sim)
    CI_95 = 1.96 * std_err

    log_N = np.log(N)
    log_mean_dev = np.log(mean_dev)

    SE_log_dev = std_err / mean_dev
    CI_log_95 = 1.96 * SE_log_dev

    slope, intercept, _, _, _ = linregress(N.to_numpy(), log_mean_dev.to_numpy())
    scaling_exponents[r] = slope

    axes[0].errorbar(N, mean_dev, yerr=CI_95, marker='o', linestyle='-', capsize=5, label=f"r={r}")
    axes[1].errorbar(log_N, log_mean_dev, yerr=CI_log_95, marker='o', linestyle='-', capsize=5, label=f"r={r}")

axes[0].set_xlabel("Population size N")
axes[0].set_ylabel(r"$E[\sup_{t \in [0,T(N)]} |X(t)/N - x(t)|]$")
axes[0].set_title("Supremum deviation")
axes[0].legend()
axes[0].grid()

axes[1].set_xlabel("Log N")
axes[1].set_ylabel(r"$\log E[\sup_{t \in [0,T(N)]} |X(t)/N - x(t)|]$")
axes[1].set_title("Log supremum deviation")
axes[1].legend()
axes[1].grid()

plt.tight_layout()
plt.show()

# Print Scaling Exponents
for r, slope in scaling_exponents.items():
    print(f"For r = {r}, estimated scaling exponent: {slope:.4f}")
