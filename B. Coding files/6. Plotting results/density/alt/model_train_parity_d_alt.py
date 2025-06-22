# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 13:04:46 2025

@author: danai
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

# --- Load and clean the PCA-transformed dataset ---
file_path = 'features_RF_MI_combined.xlsx'
data = pd.read_excel(file_path)

# Set your target column name
target_column = 'Crosslink density (mol/m3)'

# Keep only numeric columns
data_numeric = data.select_dtypes(include=[np.number])

# Add the target back if it was lost when selecting numerics
if target_column not in data_numeric.columns:
    data_numeric[target_column] = data[target_column]

# Separate features and target
X = data_numeric.drop(columns=[target_column])
y = data_numeric[target_column]

# --- Define models ---
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Support Vector Machine": SVR(),
    "Gaussian Process": GaussianProcessRegressor(kernel=RBF(), random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42, verbosity=0)
}

# --- Set up KFold ---
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# --- Prepare results and prediction storage ---
results = {model_name: {'R2': [], 'RMSE': [], 'MAE': []} for model_name in models}
all_predictions = {model_name: {'y_true': [], 'y_pred': [], 'errors': []} for model_name in models}

# --- Training and evaluation ---
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Store predictions for parity plot
        all_predictions[model_name]['y_true'].extend(y_test.tolist())
        all_predictions[model_name]['y_pred'].extend(y_pred.tolist())
        all_predictions[model_name]['errors'].extend((y_pred - y_test).tolist())

        # Metrics
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        results[model_name]['R2'].append(r2)
        results[model_name]['RMSE'].append(rmse)
        results[model_name]['MAE'].append(mae)

# --- Summarize results ---
summary = {
    'Model': [],
    'R2 Mean': [],
    'R2 Std': [],
    'RMSE Mean': [],
    'RMSE Std': [],
    'MAE Mean': [],
    'MAE Std': []
}

for model_name, scores in results.items():
    summary['Model'].append(model_name)
    summary['R2 Mean'].append(np.mean(scores['R2']))
    summary['R2 Std'].append(np.std(scores['R2']))
    summary['RMSE Mean'].append(np.mean(scores['RMSE']))
    summary['RMSE Std'].append(np.std(scores['RMSE']))
    summary['MAE Mean'].append(np.mean(scores['MAE']))
    summary['MAE Std'].append(np.std(scores['MAE']))

summary_df = pd.DataFrame(summary)
summary_df.to_excel('model_performance_summary.xlsx', index=False)

# --- Denormalization ---
original_df = pd.read_excel("Draft4_cleaned2.xlsx")
y_min = original_df[target_column].min()
y_max = original_df[target_column].max()

def denormalize(y_norm):
    return 0.5 * (y_norm + 1) * (y_max - y_min) + y_min

# --- Create plots folder ---
os.makedirs("plots", exist_ok=True)

# --- Parity Plots + Error Distributions ---
for model_name, data in all_predictions.items():
    y_true = np.array(data['y_true'])
    y_pred = np.array(data['y_pred'])
    errors = np.array(data['errors'])

    y_true = denormalize(y_true)
    y_pred = denormalize(y_pred)

    # Global R²
    r2_global = r2_score(y_true, y_pred)
    r2_cv = np.mean(results[model_name]['R2'])

    # --- Parity Plot ---
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolor='k')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label='Ideal')
    plt.xlabel("Actual Crosslink Density (mol/m³)")
    plt.ylabel("Predicted Crosslink Density (mol/m³)")
    plt.title(f"Parity Plot - {model_name}")
    plt.text(0.05, 0.95, f"CV $R^2$ = {r2_cv:.2f}\nAll $R^2$ = {r2_global:.2f}",
             transform=plt.gca().transAxes, fontsize=11, verticalalignment='top')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"plots/parity_plot_{model_name.replace(' ', '_')}.png")
    plt.close()

    # --- Error Distribution ---
    plt.figure(figsize=(6, 4))
    plt.hist(errors, bins=30, color='steelblue', edgecolor='black', alpha=0.75)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=1)
    plt.xlabel("Prediction Error (y_pred - y_true)")
    plt.ylabel("Frequency")
    plt.title(f"Error Distribution - {model_name}")
    plt.grid(axis='y', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"plots/error_distribution_{model_name.replace(' ', '_')}.png")
    plt.close()