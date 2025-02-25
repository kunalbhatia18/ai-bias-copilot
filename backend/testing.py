# Temporary script to create raw_sentiment140.csv
import pandas as pd
input_path = '../data/processed/sentiment_with_gender.csv'
output_path = '../data/processed/raw_sentiment140.csv'
df = pd.read_csv(input_path)
df_raw = df[['user', 'text', 'sentiment']].copy()
# If 'user' column is missing, generate dummy usernames
if 'user' not in df_raw.columns:
    df_raw['user'] = ['user_' + str(i) for i in range(len(df_raw))]
df_raw.to_csv(output_path, index=False)
print(f"Created {output_path} with {len(df_raw)} rows")