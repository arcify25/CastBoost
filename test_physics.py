import joblib
import pandas as pd
import numpy as np

model = joblib.load('concrete_model.pkl')

feature_order = [
    'cement', 'slag', 'flyash', 'water', 'superplasticizer', 
    'coarseagg', 'fineagg', 'age', 'water_cement_ratio', 
    'slag_cement_ratio', 'flyash_cement_ratio', 'total_binder'
]

def make_mix(water, age, cement=280.0, slag=0.0, flyash=0.0):
    return {
        'cement': cement, 'slag': slag, 'flyash': flyash,
        'water': water, 'superplasticizer': 2.5,
        'coarseagg': 1000.0, 'fineagg': 750.0, 'age': age,
        'water_cement_ratio': water / cement,
        'slag_cement_ratio': slag / cement,
        'flyash_cement_ratio': flyash / cement,
        'total_binder': cement + slag + flyash
    }

# ---------------------------------------------------------
# TEST 1: Water Test (Abrams' Law) - Strength MUST decrease
# ---------------------------------------------------------
water_range = np.linspace(120, 250, 14)  # 120, 130, 140 ... 250 kg/m³
water_samples = pd.DataFrame([make_mix(w, age=28) for w in water_range])[feature_order]
water_preds = model.predict(water_samples)

water_violations = 0
for i in range(len(water_preds) - 1):
    if water_preds[i+1] > water_preds[i]:  # Strength went UP when water went UP
        water_violations += 1

# ---------------------------------------------------------
# TEST 2: Curing Age Test - Strength MUST increase
# ---------------------------------------------------------
age_range = [1, 3, 7, 14, 28, 56, 90, 180, 365]
age_samples = pd.DataFrame([make_mix(water=160.0, age=a) for a in age_range])[feature_order]
age_preds = model.predict(age_samples)

age_violations = 0
for i in range(len(age_preds) - 1):
    if age_preds[i+1] < age_preds[i]:  # Strength went DOWN as time passed
        age_violations += 1

# ---------------------------------------------------------
# VERDICT
# ---------------------------------------------------------
print("=== PHYSICS CONSTRAINTS AUDIT ===")
print(f"Water Test (120 to 250 kg/m³) : {water_preds[0]:.2f} MPa -> {water_preds[-1]:.2f} MPa")
if water_violations == 0:
    print("  [✓] PASSED: Strength monotonically decreases with water.")
else:
    print(f"  [✗] FAILED: {water_violations} unphysical upward spikes detected.")

print(f"\nAge Test (1 to 365 Days)       : {age_preds[0]:.2f} MPa -> {age_preds[-1]:.2f} MPa")
if age_violations == 0:
    print("  [✓] PASSED: Strength monotonically increases over time.")
else:
    print(f"  [✗] FAILED: {age_violations} curing anomalies detected.")