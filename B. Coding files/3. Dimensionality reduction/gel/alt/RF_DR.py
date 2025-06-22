# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 21:24:37 2025

@author: danai
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Import dataset
df = pd.read_excel("augmented_output_clean1_final2.xlsx")

# Define target and features
target_col = "Gel fraction (%)"
X = df.drop(columns=[target_col])
y = df[target_col]

# Keep only numeric features and fill NaNs
X_numeric = X.select_dtypes(include=["number"]).fillna(0)

# Fit Random Forest model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_numeric, y)

# Get feature importances
importances = rf.feature_importances_
rf_df = pd.DataFrame({
    "Feature": X_numeric.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

# Select important features

# -- Option 1: select by threshold
rf_selected = rf_df[rf_df["Importance"] > 0.003]["Feature"].tolist()

# -- Option 2: keep top N features (alternative)
# rf_selected = rf_df.head(50)["Feature"].tolist()

# Save selected features and full feature ranking
df_selected = df[rf_selected + [target_col]]
df_selected.to_excel("features_RF.xlsx", index=False)

rf_df.to_excel("features_RF.xlsx", index=False)

# Done
print("RF feature selection complete.")
print(f"{len(rf_selected)} features selected.")
print("Full feature importance ranking saved to: features_RF.xlsx")
