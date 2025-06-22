# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 18:33:44 2025

@author: danai
"""

#This is a code that applies standard or log normalization based on feature skewness

import pandas as pd
import numpy as np
from scipy.stats import skew
import matplotlib.pyplot as plt
import os

# Load your dataset
df = pd.read_excel("SMILES_list_cleaned2.xlsx")

# Select numerical columns (excluding non-numeric ones)
num_cols = df.select_dtypes(include=[np.number]).columns

# Calculate skewness
skewness = df[num_cols].apply(skew)

# Identify non-skewed features (threshold can be adjusted)
non_skewed_cols = skewness[abs(skewness) < 0.5].index
skewed_cols = skewness[abs(skewness) >= 0.5].index

print("Non-skewed features:")
print(non_skewed_cols)

# Initialize df_normalized to avoid errors when setting values
df_normalized = df.copy()

# For skewed columns
for col in skewed_cols:
    if (df[col] > 0).all():
        df_normalized[col] = np.log1p(df[col])
        # Scale log values to [-1, 1]
        df_normalized[col] = 2 * (df_normalized[col] - df_normalized[col].min()) / (df_normalized[col].max() - df_normalized[col].min()) - 1
    else:
        df_normalized[col] = 2 * (df[col] - df[col].min()) / (df[col].max() - df[col].min()) - 1

# For non-skewed columns
for col in non_skewed_cols:
    df_normalized[col] = 2 * (df[col] - df[col].min()) / (df[col].max() - df[col].min()) - 1

# Save the normalized data
df_normalized.to_excel("normalized_SMILES_list2.xlsx", index=False)

# Create folder for histograms
output_folder = "individual_histograms22"
os.makedirs(output_folder, exist_ok=True)

# Generate and save individual histograms for the normalized data 
for col in df_normalized:
    plt.figure(figsize=(6, 4))
    plt.hist(df_normalized[col], bins=30, alpha=0.75, color='blue', edgecolor='black')
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of {col}")
    plt.grid(axis='y', alpha=0.5)

    # Save the histogram
    plt.savefig(os.path.join(output_folder, f"{col}.png"))
    plt.close()

print(f"Individual histograms saved in '{output_folder}'.")

print("Normalization complete! File saved as 'normalized_SMILES_list2.xlsx'.")
                    