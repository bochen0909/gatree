# Sample Weights Support in GATree

GATree now supports sample weights for both regression and action selection tasks. Sample weights allow you to assign different importance to different samples during training, which can be useful for handling imbalanced datasets, emphasizing certain time periods, or incorporating domain knowledge about data quality.

## Overview

Sample weights are supported in:
- **GATreeRegressor**: Weights affect the Mean Squared Error calculation in the fitness function
- **GATreeActionSelector**: Weights affect individual timestep rewards and global reward calculations

## Usage

### GATreeRegressor with Sample Weights

```python
import numpy as np
import pandas as pd
from gatree.methods.gatreeregressor import GATreeRegressor

# Create sample data
X = pd.DataFrame({
    'feature1': np.random.randn(100),
    'feature2': np.random.randn(100)
})
y = pd.Series(X['feature1'] * 2 + X['feature2'] + np.random.randn(100) * 0.1)

# Create sample weights - give more importance to first half of samples
sample_weight = np.ones(len(X))
sample_weight[:50] = 2.0  # Double weight for first 50 samples

# Train with sample weights
regressor = GATreeRegressor(random_state=42)
regressor.fit(X, y, sample_weight=sample_weight, max_iter=100)

# Make predictions
predictions = regressor.predict(X)
```

### GATreeActionSelector with Sample Weights

```python
import numpy as np
import pandas as pd
from gatree.methods.gatreeactionselector import GATreeActionSelector

# Create time series data
n_timesteps = 100
X = pd.DataFrame({
    'price': np.cumsum(np.random.randn(n_timesteps) * 0.1) + 100,
    'volume': np.random.exponential(1000, n_timesteps)
})
Y = pd.DataFrame({
    'next_return': np.random.randn(n_timesteps) * 0.02
})

# Define action space and reward function
action_space = ['buy', 'sell', 'hold']

def reward_function(state, action, y_data, timestep, previous_action):
    next_return = y_data['next_return']
    
    # Base reward based on action and market direction
    if action == 'buy':
        base_reward = next_return * 100
    elif action == 'sell':
        base_reward = -next_return * 100
    else:  # hold
        base_reward = 0
    
    # Optional: Add penalty for frequent action changes
    change_penalty = 0.0
    if previous_action is not None and previous_action != action:
        change_penalty = 1.0  # Small penalty for changing actions
    
    return base_reward - change_penalty

# Create sample weights - give more importance to later timesteps
sample_weight = np.linspace(0.5, 2.0, n_timesteps)

# Train with sample weights
selector = GATreeActionSelector(
    action_space=action_space,
    reward_function=reward_function,
    random_state=42
)
selector.fit(X, Y, sample_weight=sample_weight, max_iter=50)

# Make predictions
actions = selector.predict_actions(X)
```

## How Sample Weights Work

### In GATreeRegressor

Sample weights are passed to scikit-learn's `mean_squared_error` function, which computes a weighted MSE:

```
weighted_mse = sum(sample_weight * (y_true - y_pred)^2) / sum(sample_weight)
```

This means samples with higher weights contribute more to the fitness calculation, making the genetic algorithm prioritize fitting those samples better.

### In GATreeActionSelector

Sample weights affect the fitness calculation in two ways:

1. **Individual Rewards**: Each timestep's reward is multiplied by its sample weight before applying discounting
2. **Global Rewards**: If a global reward function is used, it receives the mean sample weight as a scaling factor

The weighted fitness calculation becomes:
```
fitness = -(sum(sample_weight[t] * discounted_reward[t]) + global_reward * mean(sample_weight)) + complexity_penalty
```

## Sample Weight Validation

Both classes validate sample weights with the following rules:

- **Length**: Must match the number of samples/timesteps
- **Non-negative**: All weights must be >= 0
- **Type**: Automatically converted to numpy array if not already

```python
# Valid weights
weights = np.array([1.0, 2.0, 1.5, 1.0])  # Different importance
weights = np.ones(100)  # Equal importance (equivalent to no weights)

# Invalid weights - will raise ValueError
weights = np.array([1.0, -1.0, 2.0])  # Negative weight
weights = np.array([1.0, 2.0])  # Wrong length for 100 samples
```

## Use Cases

### 1. Handling Imbalanced Data
Give higher weights to underrepresented samples:
```python
# For regression with rare high-value samples
sample_weight = np.where(y > threshold, 3.0, 1.0)
```

### 2. Time Series Recency
Emphasize recent observations in time series:
```python
# Linear increase in importance over time
sample_weight = np.linspace(0.5, 2.0, n_timesteps)

# Exponential decay for older samples
sample_weight = np.exp(np.linspace(-2, 0, n_timesteps))
```

### 3. Data Quality Weighting
Weight samples based on data quality or confidence:
```python
# Based on measurement uncertainty
sample_weight = 1.0 / measurement_uncertainty

# Based on data source reliability
sample_weight = source_reliability_scores
```

### 4. Domain Knowledge
Incorporate business rules or domain expertise:
```python
# Higher weight for business-critical periods
sample_weight = np.where(is_critical_period, 5.0, 1.0)

# Weight by transaction volume in financial data
sample_weight = transaction_volumes / np.mean(transaction_volumes)
```

## Performance Considerations

- Sample weights add minimal computational overhead
- Weights are validated once at the beginning of training
- Memory usage increases slightly to store weight arrays
- Training time is not significantly affected

## Compatibility

- Sample weights work with all existing GATree parameters
- Compatible with custom fitness functions (weights passed via `fitness_function_kwargs`)
- Works with parallel processing (`n_jobs > 1`)
- Integrates with discount factors and global reward functions in action selection

## Examples

See the following example files for complete demonstrations:
- `examples/sample_weight_example.py` - Comprehensive examples for both regressor and action selector
- `examples/simple_weight_test.py` - Basic functionality verification

## API Reference

### GATreeRegressor.fit()
```python
def fit(self, X, y, sample_weight=None, population_size=150, max_iter=2000, 
        mutation_probability=0.1, elite_size=1, selection_tournament_size=2, 
        fitness_function_kwargs={}):
```

### GATreeActionSelector.fit()
```python
def fit(self, X, Y, sample_weight=None, population_size=100, max_iter=1000, 
        mutation_probability=0.15, elite_size=2, selection_tournament_size=3, 
        fitness_function_kwargs={}):
```

**Parameters:**
- `sample_weight` (array-like, optional): Sample weights. If None, all samples have equal weight.

**Raises:**
- `ValueError`: If sample_weight has wrong length or contains negative values.