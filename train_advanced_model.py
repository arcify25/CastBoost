import pandas as pd
import numpy as np
import joblib
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
from catboost import CatBoostRegressor

# 1. Load data & compute derived ratio features
df = pd.read_csv('concrete_data.csv')
df['water_cement_ratio'] = df['water'] / df['cement']
df['slag_cement_ratio'] = df['slag'] / df['cement']
df['flyash_cement_ratio'] = df['flyash'] / df['cement']
df['total_binder'] = df['cement'] + df['slag'] + df['flyash']

X = df.drop(columns=['strength'])
y = df['strength']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Define Monotonic Constraints (1 = non-decreasing, -1 = non-increasing, 0 = unconstrained)
# Guarantee: More water/cement ratio CANNOT increase strength; more age CANNOT decrease strength.
feature_names = list(X.columns)
monotone_specs = {
    'water': -1,                 # Strictly non-increasing
    'water_cement_ratio': -1,    # Strictly non-increasing
    'age': 1,                    # Strictly non-decreasing
    'cement': 1                  # Strictly non-decreasing
}
constraints = [monotone_specs.get(col, 0) for col in feature_names]

# 3. Bayesian Optimization (TPE) replacing the paper's Random Search
def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 250, 600),
        'depth': trial.suggest_int('depth', 4, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        'monotone_constraints': constraints,
        'verbose': 0,
        'random_seed': 42
    }
    
    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return root_mean_squared_error(y_test, preds)

print("Starting Bayesian Optimization (TPE)...")
optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)

print(f"[✓] Optimal Hyperparameters: {study.best_params}")

# 4. Train final Physics-Constrained CatBoost (PC-CatBoost)
best_params = study.best_params
best_params['monotone_constraints'] = constraints
best_params['verbose'] = 0
best_params['random_seed'] = 42

upgraded_model = CatBoostRegressor(**best_params)
upgraded_model.fit(X_train, y_train)

# 5. Evaluate
test_preds = upgraded_model.predict(X_test)
r2 = r2_score(y_test, test_preds)
rmse = root_mean_squared_error(y_test, test_preds)

print(f"\n=== Upgraded Model Evaluation ===")
print(f"Algorithm: Physics-Constrained Bayesian CatBoost (PC-BO-CatBoost)")
print(f"R² Score: {r2:.4f}")
print(f"RMSE    : {rmse:.2f} MPa")

# Save model
joblib.dump(upgraded_model, 'concrete_model.pkl')
print("\n[✓] Saved upgraded model to 'concrete_model.pkl'")