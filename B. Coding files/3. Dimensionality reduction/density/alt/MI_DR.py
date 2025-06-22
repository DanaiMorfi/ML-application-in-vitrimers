# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 21:29:52 2025

@author: danai
"""

import pandas as pd
from sklearn.feature_selection import mutual_info_regression

# Load dataset
df = pd.read_excel("augmented_output_clean1_final2.xlsx")

# Define target column
target_col = "Crosslink density (mol/m3)"
X = df.drop(columns=[target_col])
y = df[target_col]

# Keep only numeric features and fill NaNs
X_numeric = X.select_dtypes(include=["number"]).fillna(0)

# Compute Mutual Information
mi_scores = mutual_info_regression(X_numeric, y, random_state=42)

mi_df = pd.DataFrame({
    "Feature": X_numeric.columns,
    "Mutual Info": mi_scores
}).sort_values(by="Mutual Info", ascending=False)

# Select important features

# -- Option 1: select by threshold
mi_selected = mi_df[mi_df["Mutual Info"] > 0.25]["Feature"].tolist()

# -- Option 2: keep top N features (alternative)
# mi_selected = mi_df.head(50)["Feature"].tolist()

# Save selected features and full MI ranking
df_selected = df[mi_selected + [target_col]]
df_selected.to_excel("features_MI.xlsx", index=False)

mi_df.to_excel("features_MI.xlsx", index=False)

# Done
print("MI feature selection complete.")
print(f"{len(mi_selected)} features selected.")
print("Full MI ranking saved to: features_MI.xlsx")
