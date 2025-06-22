# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 19:52:48 2025

@author: danai
"""

import pandas as pd

# List of specific columns to check for correlation
cols_to_normalize = ['MW1 (g/mol)', 'Functional groups1', 'Reactive end-groups ratio', 'MW2 (g/mol)', 'Functional groups2',
                     'MW3 (g/mol)', 'Functional groups3',
                     'Catalyst molar ratio (%mol)', 'MW (g/mol)', 'T (oC)']

# Load the Excel file
input_file = "normalized_Draft42.xlsx"  
df_cleaned = pd.read_excel(input_file, header=0)

# Drop the endpoint column completely from the full dataset
df_cleaned = df_cleaned.drop(columns=['Tm2 (oC)', 'Gel fraction (%)', 'Swelling ratio (%)','Crosslink density (mol/m3)',
'Pre-exponential factor (s-1)', 'Ea (kJ/mol)', 'Tv (oC)', 'Youngs modulus (MPa)','Tensile strength (MPa)'])

# Subset the dataframe to include only the specific columns you want to check
df_subset = df_cleaned[cols_to_normalize]

# Calculate the correlation matrix for the selected columns
correlation_matrix = df_subset.corr()

# Set a threshold for high correlation (e.g., 0.9 means highly correlated)
threshold = 0.8  # Adjust the threshold as needed

# Identify pairs of highly correlated columns and store their names
high_corr_pairs = []

for col in correlation_matrix.columns:
    for row in correlation_matrix.index:
        if abs(correlation_matrix.loc[row, col]) > threshold and col != row:
            high_corr_pairs.append((row, col))  # Store the pair (row, column)

# Print out the pairs that are highly correlated
print(f"Highly correlated pairs (Threshold = {threshold}):")
for pair in high_corr_pairs:
    print(f"{pair[0]} <-> {pair[1]}: Correlation = {correlation_matrix.loc[pair[0], pair[1]]}")

# Convert to list and drop the second column from each pair
to_drop = [col[1] for col in high_corr_pairs][1::2]  # Take every second column from the list

# Drop the columns that are highly correlated from the original DataFrame
df_cleaned = df_cleaned.drop(columns=to_drop)

# Save the final cleaned dataset
output_file = "normalized_Draft4_cleaned_final2.xlsx"  # Change this to your desired output file path
df_cleaned.to_excel(output_file, index=False)

# Print which columns were removed
print(f"\nColumns removed (due to high correlation): {to_drop}")
print(f"Cleaned data saved to: {output_file}")




