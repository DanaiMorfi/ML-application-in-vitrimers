# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 18:10:41 2025

@author: danai
"""

#This is a code that performs data clean-up and normalization for the list of
#features collected from saveral articles and experimental studies for different
#types of polymeric structures

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Load dataset
df = pd.read_excel("Draft4_new2.xlsx", header=2)

df_normalized = df.copy()  # Keep a copy of the original data

# List of columns to normalize
cols_to_normalize = ['MW1 (g/mol)', 'Functional groups1', 'Reactive end-groups ratio', 'MW2 (g/mol)', 'Functional groups2',
                     'MW3 (g/mol)', 'Functional groups3',
                     'Catalyst molar ratio (%mol)', 'MW (g/mol)', 'T (oC)', 'Tm2 (oC)', 
                     'Gel fraction (%)', 'Swelling ratio (%)', 'Crosslink density (mol/m3)', 'Tg (oC)',
                     'Pre-exponential factor (s-1)', 'Ea (kJ/mol)', 'Tv (oC)', 'Youngs modulus (MPa)',
                     'Tensile strength (MPa)']

# Convert "-" to NaN and ensure numerical conversion
df[cols_to_normalize] = df[cols_to_normalize].replace("-", np.nan).apply(pd.to_numeric, errors='coerce')

# Fill NaN values with column mean
df[cols_to_normalize] = df[cols_to_normalize].fillna(df[cols_to_normalize].mean())

# Create folder for histograms
output_folder_before = "histograms"
os.makedirs(output_folder_before, exist_ok=True)

# Generate histograms 
for col in cols_to_normalize:
    plt.figure(figsize=(6, 4))
    plt.hist(df[col], bins=30, alpha=0.75, color='blue', edgecolor='black')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {col}')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(os.path.join(output_folder_before, f"{col.replace('/', '_').replace('%', 'percent')}.png"))
    plt.close()

print(f"Histograms saved in '{output_folder_before}' folder.")

# Save cleaned data
df.to_excel("Draft4_cleaned2.xlsx", index=False)

# Min-Max Normalization to range [-1, 1]
df_normalized[cols_to_normalize] = 2 * (df[cols_to_normalize] - df[cols_to_normalize].min()) / (df[cols_to_normalize].max() - df[cols_to_normalize].min()) - 1

# Fill empty cells in Catalyst Type
df_normalized['Catalyst type'] = df_normalized['Catalyst type'].replace(['-', '', None, np.nan], 'No molec')

# Fill empty cells in Compound Type.1
df_normalized['Compound type.1'] = df_normalized['Compound type.1'].replace(['-', '', None, np.nan], 'No molec')

# Fill empty cells in Compound Type.2
df_normalized['Compound type.2'] = df_normalized['Compound type.2'].replace(['-', '', None, np.nan], 'No molec')

# Save normalized data
df_normalized.to_excel("normalized_Draft42.xlsx", index=False)

print("Normalization complete! Files saved as 'Draft4_cleaned2.xlsx' and 'normalized_Draft42.xlsx'.")

