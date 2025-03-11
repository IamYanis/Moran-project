import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress


data = pd.read_csv("data/moran_simulation_results.csv")

#extract values
N_values = sorted(data["N"].unique())
r_values = sorted(data["r"].unique())

#print("N values in dataset:", N_values)

n_sim = 1000  # This should match N_SIM in main.c

#Deviation plots
fig1, axes = plt.subplots(1, 2, figsize=(12, 5))

scaling_exponents = {}

dev_data = data[data["Type"] == "Deviation"]  

for r in r_values:
    subset = dev_data[dev_data["r"] == r]

    N = subset["N"]
    mean_dev = subset["Value"]
    variance_dev = subset["std_sup_dev"] ** 2  

    std_err = np.sqrt(variance_dev / n_sim)  
    CI_95 = 1.96 * std_err  # 95% confidence interval

    log_N = np.log(N)
    log_mean_dev = np.log(mean_dev)

    SE_log_dev = std_err / mean_dev
    CI_log_95 = 1.96 * SE_log_dev

    slope, intercept, _, _, _ = linregress(N, log_mean_dev)
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

#Print Scaling Exponents
for r, slope in scaling_exponents.items():
    print(f"For r = {r}, estimated scaling exponent: {slope:.4f}")

