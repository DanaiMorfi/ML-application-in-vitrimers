# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 22:00:19 2025

@author: danai
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import Counter

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

# --- Load dataset ---
data = pd.read_excel("pca_transformed_features.xlsx")
target_column = 'Crosslink density (mol/m3)'

# Ensure numeric columns
data_numeric = data.select_dtypes(include=[np.number])
if target_column not in data_numeric.columns:
    data_numeric[target_column] = data[target_column]

X = data_numeric.drop(columns=[target_column])
y = data_numeric[target_column]

# --- Define models and parameter grids ---
model_params = {
    "Linear Regression": {'model': LinearRegression(), 'params': {}},
    "Random Forest": {
        'model': RandomForestRegressor(random_state=42),
        'params': {'n_estimators': [100, 200], 'max_depth': [5, 10, None], 'min_samples_split': [2, 5]}
    },
    "Support Vector Machine": {
        'model': SVR(),
        'params': {'C': [0.1, 1, 10], 'epsilon': [0.01, 0.1], 'kernel': ['rbf', 'linear']}
    },
    "Gaussian Process": {
        'model': GaussianProcessRegressor(random_state=42),
        'params': {
            'kernel': [
                ConstantKernel(1.0) * RBF(10) + WhiteKernel(0.1),
                ConstantKernel(0.1) * RBF(1) + WhiteKernel(0.01)
            ]
        }
    },
    "Gradient Boosting": {
        'model': GradientBoostingRegressor(random_state=42),
        'params': {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5], 'max_features': ['auto', 'sqrt']}
    },
    "XGBoost": {
        'model': XGBRegressor(random_state=42, verbosity=0),
        'params': {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5, 7]}
    }
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = {name: {'R2': [], 'RMSE': [], 'MAE': []} for name in model_params}
all_predictions = {name: {'y_true': [], 'y_pred': [], 'errors': []} for name in model_params}
best_params_per_model = {name: [] for name in model_params}

# --- Training and evaluation ---
for train_idx, test_idx in kf.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    for name, config in model_params.items():
        grid = GridSearchCV(config['model'], config['params'], cv=3, scoring='r2', n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)

        results[name]['R2'].append(r2_score(y_test, y_pred))
        results[name]['RMSE'].append(np.sqrt(mean_squared_error(y_test, y_pred)))
        results[name]['MAE'].append(mean_absolute_error(y_test, y_pred))

        all_predictions[name]['y_true'].extend(y_test.tolist())
        all_predictions[name]['y_pred'].extend(y_pred.tolist())
        all_predictions[name]['errors'].extend((y_pred - y_test).tolist())

        best_params_per_model[name].append(grid.best_params_)

# --- Create results summary ---
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
summary_df.to_excel("model_performance_summary_CV.xlsx", index=False)

# --- Save detailed and summary hyperparameters ---
param_details = {'Model': [], 'Fold': [], 'Best Params': []}
param_summary = {'Model': [], 'Most Common Params': []}

for model_name, params_list in best_params_per_model.items():
    for i, params in enumerate(params_list):
        param_details['Model'].append(model_name)
        param_details['Fold'].append(f'Fold {i+1}')
        param_details['Best Params'].append(str(params))
    
    try:
        # Get most common param combination (mode)
        most_common = Counter([str(p) for p in params_list]).most_common(1)[0][0]
        param_summary['Model'].append(model_name)
        param_summary['Most Common Params'].append(most_common)
    except:
        param_summary['Model'].append(model_name)
        param_summary['Most Common Params'].append("N/A")

param_details_df = pd.DataFrame(param_details)
param_summary_df = pd.DataFrame(param_summary)

with pd.ExcelWriter("model_best_hyperparameters_CV.xlsx") as writer:
    param_details_df.to_excel(writer, sheet_name='Details', index=False)
    param_summary_df.to_excel(writer, sheet_name='Summary', index=False)

# --- Denormalize ---
original_df = pd.read_excel("Draft4_cleaned2.xlsx")
y_min = original_df[target_column].min()
y_max = original_df[target_column].max()

def denormalize(y_norm, y_min, y_max):
    return 0.5 * (np.array(y_norm) + 1) * (y_max - y_min) + y_min

# --- Plotting ---
os.makedirs("plots", exist_ok=True)

for model_name, data in all_predictions.items():
    y_true = denormalize(data['y_true'], y_min, y_max)
    y_pred = denormalize(data['y_pred'], y_min, y_max)
    errors = y_pred - y_true

    # Parity plot
    r2_global = r2_score(y_true, y_pred)
    r2_cv = np.mean(results[model_name]['R2'])

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolor='k')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("Actual Crosslink Density (mol/m³)")
    plt.ylabel("Predicted Crosslink Density (mol/m³)")
    plt.title(f"Parity Plot - {model_name}")
    plt.text(0.05, 0.95, f"CV $R^2$ = {r2_cv:.2f}\nAll $R^2$ = {r2_global:.2f}",
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"plots/parity_plot_{model_name.replace(' ', '_')}.png")
    plt.close()

    # Error distribution
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

print("✓ Parity and error distribution plots saved.")
print("✓ Summary saved as Excel.")
print("✓ Best hyperparameters (details & summary) saved to Excel.")






