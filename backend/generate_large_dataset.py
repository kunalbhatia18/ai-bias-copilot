# generate_large_dataset.py
import pandas as pd

df = pd.read_csv('../data/processed/raw_sentiment140.csv')
large_df = pd.concat([df] * 5, ignore_index=True)
large_df.to_csv('large_dataset.csv', index=False)
print(f"Generated large_dataset.csv with {len(large_df)} rows")