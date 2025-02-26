# test_empty.py
from bias_copilot import mitigate_bias
mitigated_model, metrics = mitigate_bias(None, '../empty.csv')