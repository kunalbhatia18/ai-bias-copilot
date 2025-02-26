from bias_copilot import BiasAnalyzer
import pandas as pd
import time

start_time = time.time()
data = pd.read_csv('large_dataset.csv')
results = BiasAnalyzer().analyze(data)
end_time = time.time()
print("Before:", results['before'])
print("After:", results['after'])
print(f"Runtime: {end_time - start_time:.2f} seconds")