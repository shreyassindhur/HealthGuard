import pandas as pd

df = pd.read_csv('resources/dataset_kaggle.csv')
print(f'Dataset has {len(df)} rows')
print(f'Unique diseases: {df["Disease"].nunique()}')
print('\nAll diseases:')
for i, disease in enumerate(df['Disease'].unique()):
    print(f"{i}: {disease}")
