# GATree Continuous Action Selection

## Overview

The `GATreeContinuousActionSelector` extends the GATree framework to handle **continuous action spaces** for sequential decision-making problems. Unlike the discrete `GATreeActionSelector` that chooses from a predefined set of actions, the continuous version outputs real-valued actions within specified bounds.

## Key Differences from Discrete Action Selection

| Feature | GATreeActionSelector | GATreeContinuousActionSelector |
|---------|---------------------|--------------------------------|
| **Action Space** | Discrete (e.g., ['buy', 'sell', 'hold']) | Continuous (e.g., 0.0 to 1.0) |
| **Output Type** | String/categorical actions | Float values within bounds |
| **Internal Representation** | Direct action indices | Discretized bins mapped to continuous values |
| **Use Cases** | Classification-like decisions | Regression-like control problems |

## Architecture

### Continuous Action Mapping

The continuous action selector uses an internal discretization approach:

1. **Action Bounds**: Define the continuous range (e.g., `(0, 1)`)
2. **Action Bins**: Discretize the range into `n_action_bins` (default: 50)
3. **Tree Output**: Decision tree outputs discrete bin indices
4. **Continuous Mapping**: Bin indices are mapped to continuous values

```python
# Example: 10 bins for range (0, 1)
# Bin 0 → 0.0, Bin 1 → 0.111, ..., Bin 9 → 1.0
continuous_action = min_bound + (bin_index / (n_bins - 1)) * (max_bound - min_bound)
```

### Fitness Function

The fitness function evaluates continuous actions by:

1. **Simulating Action Sequence**: Convert tree predictions to continuous actions
2. **Calculating Rewards**: Apply reward function to each (state, continuous_action) pair
3. **Applying Discounting**: Support forward/backward discounting
4. **Adding Penalties**: Complexity penalty and optional global rewards

## Usage Examples

### Basic Portfolio Allocation

```python
from gatree.methods.gatreecontinuousactionselector import GATreeContinuousActionSelector

def portfolio_reward(state, action, y_data, timestep, previous_action):
    """
    Reward function for portfolio allocation.
    action: 0.0 = all cash, 1.0 = all stocks
    """
    market_return = y_data['market_return']
    risk_free_rate = y_data['risk_free_rate']
    
    # Portfolio return
    portfolio_return = action * market_return + (1 - action) * risk_free_rate
    
    # Risk penalty
    risk_penalty = 0.1 * (action ** 2) * y_data['volatility']
    
    return portfolio_return - risk_penalty

# Create selector
selector = GATreeContinuousActionSelector(
    reward_function=portfolio_reward,
    action_bounds=(0, 1),  # 0% to 100% stock allocation
    n_action_bins=50,      # Fine-grained control
    discount_factor=0.99,
    random_state=42
)

# Train
selector.fit(X_train, Y_train, population_size=50, max_iter=100)

# Predict continuous actions
actions = selector.predict_actions(X_test)  # Returns [0.23, 0.67, 0.45, ...]
```

### Custom Action Bounds

```python
# Temperature control: -10°C to +10°C
selector = GATreeContinuousActionSelector(
    reward_function=temperature_reward,
    action_bounds=(-10, 10),
    n_action_bins=100,  # 0.2°C precision
    random_state=42
)

# Speed control: 0 to 100 km/h
selector = GATreeContinuousActionSelector(
    reward_function=speed_reward,
    action_bounds=(0, 100),
    n_action_bins=200,  # 0.5 km/h precision
    random_state=42
)
```

## Key Parameters

### Action Space Configuration

- **`action_bounds`**: `(min, max)` tuple defining the continuous range
- **`n_action_bins`**: Number of discrete bins for internal representation
  - Higher values = finer control but larger search space
  - Recommended: 20-100 depending on precision needs

### Reward Function Signature

```python
def reward_function(state, action, y_data, timestep, previous_action):
    """
    Args:
        state (pd.Series): Current state features
        action (float): Continuous action value within bounds
        y_data (pd.Series): Additional data for reward calculation
        timestep (int): Current time step
        previous_action (float or None): Previous continuous action
    
    Returns:
        float: Reward value
    """
```

### Training Parameters

- **`discount_factor`**: Future reward discounting (0.0 to 1.0)
- **`discount_direction`**: `'forward'` (standard) or `'backward'` (reverse)
- **`global_reward_function`**: Optional function for sequence-level rewards
- **`sample_weight`**: Per-timestep importance weights

## Advanced Features

### Global Reward Functions

```python
def sharpe_ratio_reward(rewards, actions, X, Y):
    """Global reward based on risk-adjusted returns."""
    if len(rewards) < 2:
        return 0.0
    
    mean_return = np.mean(rewards)
    std_return = np.std(rewards)
    
    return (mean_return / std_return) if std_return > 0 else 0.0

selector = GATreeContinuousActionSelector(
    reward_function=base_reward,
    global_reward_function=sharpe_ratio_reward,
    global_reward_weight=0.2,  # 20% weight on global component
    action_bounds=(0, 1)
)
```

### Backward Discounting

```python
# Emphasize later rewards (useful for goal-reaching tasks)
selector = GATreeContinuousActionSelector(
    reward_function=goal_reward,
    discount_factor=0.9,
    discount_direction='backward',  # Later rewards worth more
    action_bounds=(0, 1)
)
```

### Sample Weights

```python
# Weight later timesteps more heavily
sample_weights = np.linspace(0.5, 2.0, len(X_train))

selector.fit(
    X_train, Y_train,
    sample_weight=sample_weights,
    population_size=50,
    max_iter=100
)
```

## Performance Considerations

### Action Bin Selection

- **Too few bins** (< 10): Limited precision, may miss optimal actions
- **Too many bins** (> 200): Large search space, slower convergence
- **Sweet spot**: 20-100 bins for most applications

### Training Efficiency

- **Population size**: 30-100 for most problems
- **Max iterations**: 50-200 depending on complexity
- **Early stopping**: Recommended with patience=20-50

### Memory Usage

Memory scales with:
- Population size × Tree size × Number of features
- Continuous action selector has similar memory footprint to discrete version

## Comparison with Other Methods

### vs. Reinforcement Learning

| Aspect | GATree Continuous | Deep RL (PPO/SAC) |
|--------|------------------|-------------------|
| **Interpretability** | High (decision tree) | Low (neural network) |
| **Sample Efficiency** | Medium | Low-Medium |
| **Hyperparameter Tuning** | Moderate | High |
| **Continuous Actions** | Discretized | Native |
| **Training Time** | Medium | High |

### vs. Traditional Control

| Aspect | GATree Continuous | PID/MPC |
|--------|------------------|---------|
| **Adaptability** | High (learns from data) | Low (manual tuning) |
| **Nonlinearity** | High | Medium |
| **Interpretability** | High | High |
| **Real-time Performance** | Fast | Very Fast |

## Best Practices

### 1. Reward Function Design

```python
def good_reward_function(state, action, y_data, timestep, previous_action):
    # ✅ Scale rewards to reasonable range (-10 to +10)
    # ✅ Include action change penalties for stability
    # ✅ Handle edge cases gracefully
    
    try:
        base_reward = calculate_base_reward(state, action, y_data)
        
        # Stability penalty
        if previous_action is not None:
            stability_penalty = 0.01 * abs(action - previous_action)
            base_reward -= stability_penalty
        
        return np.clip(base_reward, -10, 10)  # Prevent extreme values
    except:
        return -1.0  # Safe fallback
```

### 2. Action Bounds Selection

```python
# ✅ Choose meaningful bounds based on problem domain
action_bounds = (0, 1)      # Portfolio allocation (0% to 100%)
action_bounds = (-1, 1)     # Normalized control signal
action_bounds = (0, 100)    # Speed control (0 to 100 km/h)

# ❌ Avoid arbitrary or too-wide bounds
action_bounds = (-1000, 1000)  # Too wide, hard to optimize
```

### 3. Training Configuration

```python
# ✅ Recommended configuration for most problems
selector.fit(
    X, Y,
    population_size=50,        # Balance between diversity and speed
    max_iter=100,             # Usually sufficient with early stopping
    mutation_probability=0.15, # Moderate exploration
    elite_size=5,             # Preserve best solutions
    early_stopping=True,      # Prevent overfitting
    patience=20               # Allow for convergence
)
```

## Troubleshooting

### Common Issues

1. **All actions are the same**: Increase `n_action_bins` or check reward function
2. **Slow convergence**: Reduce population size or increase mutation probability
3. **Unstable actions**: Add action change penalties in reward function
4. **Poor performance**: Check reward function scaling and bounds

### Debugging Tips

```python
# Check action distribution
stats = selector.get_action_statistics(X_test)
print(f"Action range: {stats['min']:.3f} to {stats['max']:.3f}")
print(f"Action std: {stats['std']:.3f}")

# Visualize training progress
selector.plot_fitness_evolution()

# Analyze action sequence
selector.plot_action_sequence(X_test)
```

## Examples

See `examples/continuous_action_example.py` for a complete portfolio allocation example demonstrating:

- Synthetic market data generation
- Continuous reward function design
- Model training and evaluation
- Performance comparison with baseline strategies
- Visualization of results

The example shows how GATree continuous action selection can outperform simple fixed-allocation strategies in a simulated trading environment.