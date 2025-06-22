# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 18:00:16 2025

@author: danai
"""

import pandas as pd

# === 1. Load your Excel files ===
draft_path = "normalized_Draft42.xlsx"
smiles_path = "normalized_SMILES_list2.xlsx"

draft_df = pd.read_excel(draft_path)
smiles_df = pd.read_excel(smiles_path)

# === 2. Identify target columns (compound and catalyst types) ===
target_columns = [col for col in draft_df.columns if col.startswith(("Compound type", "Catalyst type"))]

# === 3. Prepare SMILES data ===
smiles_df_unique = (
    smiles_df
    .drop_duplicates(subset="CORRESPONDING MOLECULE")
    .set_index("CORRESPONDING MOLECULE")
    .drop(columns=["SMILES"], errors="ignore")  # Drop SMILES column if it exists
)

smiles_features = smiles_df_unique.columns

# === 4. Start with a clean copy of the draft ===
augmented_df = draft_df.copy()

# === 5. For each target column, add all SMILES features next to it ===
for col in reversed(target_columns):  # reversed to avoid index shift when inserting
    suffix = col.split()[-1]
    compound_series = augmented_df[col]

    # Create all new columns at once using mapping
    new_data = pd.DataFrame({
        f"{feature}{suffix}": compound_series.map(smiles_df_unique[feature])
        for feature in smiles_features
    })

    # Determine insertion point
    insert_at = augmented_df.columns.get_loc(col) + 1
    left = augmented_df.iloc[:, :insert_at]
    right = augmented_df.iloc[:, insert_at:]

    # Concatenate all parts
    augmented_df = pd.concat([left, new_data, right], axis=1)

# === 6. Drop endpoint columns completely ===
augmented_df = augmented_df.drop(columns=[
    'Tm2 (oC)', 'Gel fraction (%)', 'Swelling ratio (%)', 'Crosslink density (mol/m3)',
    'Pre-exponential factor (s-1)', 'Ea (kJ/mol)', 'Tv (oC)',
    'Youngs modulus (MPa)', 'Tensile strength (MPa)'
], errors='ignore')  # Ignores any missing columns

# === 7. Save to Excel ===
augmented_df.to_excel("augmented_output_clean.xlsx", index=False)

print("Augmentation complete! SMILES features added and endpoint columns removed.")








