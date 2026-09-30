import pandas as pd
from ucimlrepo import fetch_ucirepo

# Fetch Concrete Compressive Strength dataset (UCI ID = 165)
concrete = fetch_ucirepo(id=165)
df = pd.concat([concrete.data.features, concrete.data.targets], axis=1)

# Rename to clean, lowercase column names
df.columns = [
    'cement', 'slag', 'flyash', 'water',
    'superplasticizer', 'coarseagg', 'fineagg', 'age', 'strength'
]

df.to_csv('concrete_data.csv', index=False)
print("Saved 1030 rows to concrete_data.csv successfully!")
