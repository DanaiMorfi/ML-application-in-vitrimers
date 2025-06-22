# -*- coding: utf-8 -*-
"""
Created on Sat May 17 22:16:16 2025

@author: danai
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'augmented_output_clean.xlsx'
data = pd.read_excel(file_path)

# Target column
target_column = 'Gel fraction (%)'

# Keep only numeric features
data_numeric = data.select_dtypes(include=[np.number])

# Add the target column back if needed
if target_column not in data_numeric.columns:
    data_numeric[target_column] = data[target_column]

# Separate features and target
X = data_numeric.drop(columns=[target_column])
y = data_numeric[target_column]

# Apply PCA (data already normalized)
pca = PCA()
X_pca = pca.fit_transform(X)

# Save top N PCs (e.g., 4) + target to Excel
n_components = 4
pca_df = pd.DataFrame(X_pca[:, :n_components], columns=[f'PC{i+1}' for i in range(n_components)])
pca_df[target_column] = y.values
pca_df.to_excel('pca_transformed_features.xlsx', index=False)

# === PCA Loadings Heatmap ===
loadings = pd.DataFrame(
    pca.components_[:n_components].T,  # rows = features, cols = PCs
    index=X.columns,
    columns=[f'PC{i+1}' for i in range(n_components)]
)

# Optionally: limit to top contributing features for visualization
top_features = (
    loadings
    .abs()
    .sum(axis=1)
    .sort_values(ascending=False)
    .head(30)
    .index
)

loadings_filtered = loadings.loc[top_features]

# Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(loadings_filtered, cmap='vlag', center=0, annot=True, fmt=".2f", linewidths=0.5)
plt.title("PCA Loadings Heatmap (Top 30 Features)")
plt.xlabel("Principal Components")
plt.ylabel("Features")
plt.tight_layout()
plt.savefig("pca_loadings_heatmap.png")
plt.show()


