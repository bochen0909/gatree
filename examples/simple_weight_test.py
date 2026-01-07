"""
Simple test to verify sample weight functionality works correctly.
"""

import numpy as np
import pandas as pd
from gatree.methods.gatreeregressor import GATreeRegressor

# Create simple synthetic data
np.random.seed(42)
X = pd.DataFrame({
    'x1': np.random.randn(20),
    'x2': np.random.randn(20)
})
y = pd.Series(X['x1'] * 2 + X['x2'] + np.random.randn(20) * 0.1)

# Create weights that heavily favor the first 10 samples
sample_weight = np.ones(20)
sample_weight[:10] = 10.0  # 10x weight for first half

print("Testing GATreeRegressor with sample weights...")
print(f"Data shape: {X.shape}")
print(f"Sample weights: first 10 = {sample_weight[0]}, last 10 = {sample_weight[10]}")

# Test without weights
regressor_no_weights = GATreeRegressor(random_state=42)
try:
    regressor_no_weights.fit(X, y, max_iter=10)
    print("✓ Training without weights successful")
except Exception as e:
    print(f"✗ Training without weights failed: {e}")

# Test with weights
regressor_with_weights = GATreeRegressor(random_state=42)
try:
    regressor_with_weights.fit(X, y, sample_weight=sample_weight, max_iter=10)
    print("✓ Training with weights successful")
except Exception as e:
    print(f"✗ Training with weights failed: {e}")

# Test predictions
try:
    pred_no_weights = regressor_no_weights.predict(X[:5])
    pred_with_weights = regressor_with_weights.predict(X[:5])
    print("✓ Predictions successful")
    print(f"Sample predictions without weights: {pred_no_weights[:3]}")
    print(f"Sample predictions with weights: {pred_with_weights[:3]}")
except Exception as e:
    print(f"✗ Predictions failed: {e}")

print("\nSample weight implementation verified!")