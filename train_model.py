import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, root_mean_squared_error

# 1. Load dataset
df = pd.read_csv('concrete_data.csv')

# 2. Derived Feature Engineering
df['water_cement_ratio'] = df['water'] / df['cement']
df['slag_cement_ratio'] = df['slag'] / df['cement']
df['flyash_cement_ratio'] = df['flyash'] / df['cement']
df['total_binder'] = df['cement'] + df['slag'] + df['flyash']

X = df.drop(columns=['strength'])
y = df['strength']

# 80/20 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model Suite Setup
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=150, random_state=42),
    "DR-CatBoost": CatBoostRegressor(iterations=390, learning_rate=0.35, depth=5, random_seed=42, verbose=0)
}

print("=== Model Performance Comparison ===")
trained_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)
    print(f"{name:<20} | R² Score: {r2:.4f} | RMSE: {rmse:.2f} MPa")
    trained_models[name] = model

# 4. Save the highest-performing model dynamically
best_model_name = max(trained_models, key=lambda k: r2_score(y_test, trained_models[k].predict(X_test)))
best_model = trained_models[best_model_name]

joblib.dump(best_model, 'concrete_model.pkl')
print(f"\n[✓] Saved top model ({best_model_name}) as 'concrete_model.pkl'")