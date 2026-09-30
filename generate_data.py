import pandas as pd

# Direct link to the authentic UCI Concrete Compressive Strength dataset
url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/concrete.csv"

# Load the real dataset
df = pd.read_csv(url)

# Rename columns to match our project pipeline exactly
df.columns = [
    'cement', 'slag', 'flyash', 'water', 
    'superplasticizer', 'coarseagg', 'fineagg', 'age', 'strength'
]

# Overwrite the synthetic concrete_data.csv with the real 1,030 laboratory records
df.to_csv('concrete_data.csv', index=False)
print(f"[✓] Successfully downloaded {len(df)} real laboratory records into 'concrete_data.csv'!")