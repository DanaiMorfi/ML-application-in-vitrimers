# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 22:00:19 2025

@author: danai
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

# --- Load and clean the dataset ---
file_path = 'features_RF_MI_combined.xlsx'
data = pd.read_excel(file_path)

# Set your target column name
target_column = 'Tv (oC)'
data_numeric = data.select_dtypes(include=[np.number])
if target_column not in data_numeric.columns:
    data_numeric[target_column] = data[target_column]

X = data_numeric.drop(columns=[target_column])
y = data_numeric[target_column]

# --- Define models and their hyperparameter grids ---
model_params = {
    "Linear Regression": {
        'model': LinearRegression(),
        'params': {}  # No hyperparameters to tune
    },
    "Random Forest": {
        'model': RandomForestRegressor(random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5]
        }
    },
    "Support Vector Machine": {
        'model': SVR(),
        'params': {
            'C': [0.1, 1, 10],
            'epsilon': [0.01, 0.1],
            'kernel': ['rbf', 'linear']
        }
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
        'params': {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5],
            'max_features': ['auto', 'sqrt']
        }
    },
    "XGBoost": {
        'model': XGBRegressor(random_state=42, verbosity=0),
        'params': {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5, 7]
        }
    }
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = {name: {'R2': [], 'RMSE': [], 'MAE': []} for name in model_params}

# --- Cross-validation with grid search ---
for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
    print(f"\n--- Fold {fold} ---")
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    for name, config in model_params.items():
        print(f"\n{name}")
        grid = GridSearchCV(config['model'], config['params'], cv=3, scoring='r2', n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)

        # Store metrics
        results[name]['R2'].append(r2_score(y_test, y_pred))
        results[name]['RMSE'].append(np.sqrt(mean_squared_error(y_test, y_pred)))
        results[name]['MAE'].append(mean_absolute_error(y_test, y_pred))

        print("Best Params:", grid.best_params_)

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
print("\nFinal Performance Summary:")
print(summary_df)
summary_df.to_excel('model_performance_summary_CV.xlsx', index=False)




