# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 19:40:27 2025

@author: danai
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Load dataset
df = pd.read_excel("SMILES_list_with_descriptors.xlsx", header=0)

df_normalized = df.copy()  # Keep a copy of the original data

# Select columns from index 3 to end (Python indexing: starts from 0)
cols_to_normalize = df.columns[2:] # 2: means selecting columns 3 and onwards

# Convert "-" to NaN and ensure numerical conversion
df[cols_to_normalize] = df[cols_to_normalize].replace("-", np.nan).apply(pd.to_numeric, errors='coerce')

# Fill NaN values with 0 to show absence of value
df[cols_to_normalize] = df[cols_to_normalize].fillna(0)

# Calculate the standard deviation of each column
std_dev = df[cols_to_normalize].std()
print(std_dev)

# Remove columns with zero variance
zero_variance_cols = std_dev[std_dev == 0].index.tolist()

# Remove columns with low variance based on a strict threshold
range_values = df[cols_to_normalize].max() - df[cols_to_normalize].min()
threshold_fraction = 0.000001 
thresholds = range_values * threshold_fraction

# Identify columns with standard deviation lower than the stricter threshold
low_variation_cols = std_dev[std_dev < thresholds].index.tolist() 

sparse_cols = []
for col in cols_to_normalize:
    if df[col].value_counts(normalize=True).get(0, 0) > 0.9:
        sparse_cols.append(col)

columns_to_drop = list(set(zero_variance_cols + low_variation_cols + sparse_cols))

# Drop columns with low or zero variance or full of outliers
df_cleaned = df.drop(columns=columns_to_drop)

# Optionally, save the cleaned data without low variance features
df_cleaned.to_excel("SMILES_list_cleaned2.xlsx", index=False)
print(f"Columns with low or zero variance removed: {columns_to_drop}")

# Create folder for histograms
output_folder = "individual_histograms2"
os.makedirs(output_folder, exist_ok=True)

# Generate and save individual histograms for the cleaned data (after removing low-variance columns)
for col in df_cleaned.columns:
    plt.figure(figsize=(6, 4))
    plt.hist(df_cleaned[col], bins=30, alpha=0.75, color='blue', edgecolor='black')
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of {col}")
    plt.grid(axis='y', alpha=0.5)

    # Save the histogram
    plt.savefig(os.path.join(output_folder, f"{col}.png"))
    plt.close()

print(f"Individual histograms saved in '{output_folder}'.")

print("Normalization complete! File saved as 'SMILES_list_cleaned2.")

