"""
Plot baseline vs. neighborhood-context Pearson correlation, grouped by task,
with error bars from the std across folds. Numbers are hardcoded from the
runs already recorded in the README - this is just for the summary figure,
not a live re-computation.

Usage:
    python plot_results.py
"""
import matplotlib.pyplot as plt
import numpy as np

tasks = ["PRAD\n(2 patients)", "ccRCC\n(24 patients)"]
baseline_mean = [0.3666, 0.2526]
baseline_std = [0.0089, 0.0690]
context_mean = [0.3716, 0.2854]
context_std = [0.0059, 0.0696]

x = np.arange(len(tasks))
width = 0.35

fig, ax = plt.subplots(figsize=(7, 5))

bars1 = ax.bar(x - width / 2, baseline_mean, width, yerr=baseline_std,
               label="baseline (single patch)", capsize=5, color="#8899aa")
bars2 = ax.bar(x + width / 2, context_mean, width, yerr=context_std,
               label="+ neighborhood context", capsize=5, color="#3a6ea5")

ax.set_ylabel("Mean Pearson correlation")
ax.set_title("Neighborhood context vs. single-patch baseline")
ax.set_xticks(x)
ax.set_xticklabels(tasks)
ax.legend()
ax.set_ylim(0, 0.45)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

for bars in (bars1, bars2):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.3f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 8), textcoords="offset points",
                   ha="center", fontsize=9)

plt.tight_layout()
plt.savefig("results_plot.png", dpi=150)
print("saved results_plot.png")
