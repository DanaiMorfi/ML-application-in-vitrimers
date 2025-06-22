# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 13:41:54 2025

@author: danai
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_excel("model_performance_summary_CV.xlsx")

# Remove models with very poor performance (e.g., negative R²)
df = df[df['R2 Mean'] >= 0]

# Metrics and colors
metrics = ['R2 Mean', 'RMSE Mean', 'MAE Mean']
stds = ['R2 Std', 'RMSE Std', 'MAE Std']
colors = ['#66c2a5', '#fc8d62', '#8da0cb']
labels = ['R²', 'RMSE', 'MAE']

# Bar setup
x = np.arange(len(df['Model']))  # model positions
width = 0.25  # width of each bar

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

for i in range(len(metrics)):
    ax.bar(x + i*width, df[metrics[i]], width, yerr=df[stds[i]], 
           capsize=5, label=labels[i], color=colors[i], edgecolor='black')

# Labels and formatting
ax.set_xticks(x + width)
ax.set_xticklabels(df['Model'], rotation=45, ha='right')
ax.set_ylabel("Score / Error Value")
ax.set_title("Model Performance Comparison (Mean ± Std)")
ax.legend()
ax.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig("grouped_model_comparison.png")
plt.show()
