# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 16:31:51 2025

@author: danai
"""

import pandas as pd

# Load the Excel file
input_file = "augmented_output_clean12.xlsx"
df_cleaned = pd.read_excel(input_file)

# Define columns to keep regardless of correlation
protected_columns = [
    "Reactive end-groups ratio",
    "Functional groups1"
    "Functional groups2"
    "Catalyst molar ratio (%mol)",
    "Tv (oC)"
]

# Select numeric columns
numeric_df = df_cleaned.select_dtypes(include=["number"])

# Exclude protected columns from correlation analysis
correlation_df = numeric_df.drop(columns=[col for col in protected_columns if col in numeric_df.columns])

# Calculate correlation matrix
correlation_matrix = correlation_df.corr()

# Set correlation threshold
threshold = 0.8

# Identify highly correlated pairs
high_corr_pairs = []
for col in correlation_matrix.columns:
    for row in correlation_matrix.index:
        if col != row and abs(correlation_matrix.loc[row, col]) > threshold:
            high_corr_pairs.append((row, col))

# Remove reversed duplicates
unique_pairs = set()
for a, b in high_corr_pairs:
    if (b, a) not in unique_pairs:
        unique_pairs.add((a, b))

# Print highly correlated pairs
print(f"Highly correlated pairs (Threshold = {threshold}):")
for a, b in unique_pairs:
    print(f"{a} <-> {b}: Correlation = {correlation_matrix.loc[a, b]}")

# Decide which columns to drop (excluding protected ones)
to_drop = [b for a, b in list(unique_pairs)[1::2] if b not in protected_columns]

# Drop them from the original DataFrame
df_cleaned = df_cleaned.drop(columns=to_drop)

# Save the result
output_file = "augmented_output_clean1_final2.xlsx"
df_cleaned.to_excel(output_file, index=False)

print(f"Columns removed (due to high correlation): {to_drop}")
print(f"Cleaned data saved to: {output_file}")
