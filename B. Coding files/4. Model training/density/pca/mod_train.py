# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 22:00:19 2025

@author: danai
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

# --- Load and clean the dataset ---
file_path = 'pca_transformed_features.xlsx'
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

# --- Prepare results storage ---
results = {model_name: {'R2': [], 'RMSE': [], 'MAE': []} for model_name in models}

# --- Training and evaluation ---
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

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

# --- Display final table ---
print(summary_df)

# Optional: Save to Excel
summary_df.to_excel('model_performance_summary.xlsx', index=False)



