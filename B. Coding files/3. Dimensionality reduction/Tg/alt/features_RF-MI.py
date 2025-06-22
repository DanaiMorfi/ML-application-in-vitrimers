# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 21:43:47 2025

@author: danai
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression

# Load dataset
df = pd.read_excel("augmented_output_clean1_final2.xlsx")

# Define target and features
target_col = "Tg (oC)"
X = df.drop(columns=[target_col])
y = df[target_col]

# Keep only numeric features and fill NaNs
X_numeric = X.select_dtypes(include=["number"]).fillna(0)

# Fit Random Forest model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_numeric, y)

# Get RF feature importances
rf_importances = rf.feature_importances_
rf_df = pd.DataFrame({
    "Feature": X_numeric.columns,
    "Importance": rf_importances
}).sort_values(by="Importance", ascending=False)

# Select important features by RF threshold
rf_selected = rf_df[rf_df["Importance"] > 0.008]["Feature"].tolist()

# Compute Mutual Information
mi_scores = mutual_info_regression(X_numeric, y, random_state=42)
mi_df = pd.DataFrame({
    "Feature": X_numeric.columns,
    "Mutual Info": mi_scores
}).sort_values(by="Mutual Info", ascending=False)

# Select important features by MI threshold
mi_selected = mi_df[mi_df["Mutual Info"] > 0.5]["Feature"].tolist()

# Keep only features important in both methods
selected_features = list(set(rf_selected) & set(mi_selected))

# -------------------------------
# NEW: Correlation check with target
# -------------------------------
correlation_threshold = 0.95
corr_with_target = df[selected_features + [target_col]].corr()[target_col].drop(target_col)
highly_correlated = corr_with_target[abs(corr_with_target) >= correlation_threshold]

if not highly_correlated.empty:
    print("\n Warning: The following selected features are highly correlated with the target (possible leakage?):")
    for feature, corr in highly_correlated.items():
        print(f"- {feature}: correlation = {corr:.4f}")
else:
    print("\n No selected features are highly correlated (≥ 0.95) with the target.")

# -------------------------------
# Drop only the unnecessary numeric columns
# -------------------------------
df_final = df.copy()
numeric_cols = X_numeric.columns.tolist()
cols_to_drop = [col for col in numeric_cols if col not in selected_features]
df_final = df_final.drop(columns=cols_to_drop)

# Save cleaned dataset
df_final.to_excel("features_RF_MI_combined.xlsx", index=False)

# Save full rankings
rf_df.to_excel("features_RF_full.xlsx", index=False)
mi_df.to_excel("features_MI_full.xlsx", index=False)

# Done
print("\n Combined RF and MI feature selection complete.")
print(f"{len(selected_features)} features selected.\n")
print("Selected features:")
for feature in selected_features:
    print("-", feature)

print("\n Full RF and MI rankings saved to: features_RF_full.xlsx and features_MI_full.xlsx")
print(f" Final dataset saved to features_RF_MI_combined.xlsx")

