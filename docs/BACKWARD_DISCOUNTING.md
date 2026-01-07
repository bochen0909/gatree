# Backward Discounting in GATreeActionSelector

## Overview

The GATreeActionSelector now supports **backward discounting** (reverse discounting), which prioritizes later timesteps over earlier ones. This is the opposite of standard forward discounting and is useful for scenarios where end-game performance matters most.

## Feature Description

### Forward Discounting (Standard)
```python
# Standard discounting: earlier rewards worth more
discounted_reward = reward * (discount_factor ** timestep)

# Timestep 0: weight = 1.0
# Timestep 1: weight = 0.9  
# Timestep 2: weight = 0.81
# ...
```

### Backward Discounting (New)
```python
# Reverse discounting: later rewards worth more  
reverse_timestep = n_timesteps - 1 - timestep
discounted_reward = reward * (discount_factor ** reverse_timestep)

# Timestep 0: weight = 0.81 (if n_timesteps=3)
# Timestep 1: weight = 0.9
# Timestep 2: weight = 1.0
# ...
```

## Usage

### Basic Usage
```python
from gatree.methods.gatreeactionselector import GATreeActionSelector

# Forward discounting (default)
forward_selector = GATreeActionSelector(
    action_space=[0, 1, 2],
    reward_function=my_reward_function,
    discount_factor=0.9,
    discount_direction='forward'  # Default
)

# Backward discounting
backward_selector = GATreeActionSelector(
    action_space=[0, 1, 2], 
    reward_function=my_reward_function,
    discount_factor=0.9,
    discount_direction='backward'  # New option
)
```

### Parameter Validation
```python
# Invalid direction raises ValueError
selector = GATreeActionSelector(
    action_space=[0, 1, 2],
    reward_function=my_reward_function,
    discount_direction='invalid'  # Raises ValueError
)
```

## When to Use Each Direction

### Forward Discounting (Standard)
**Use when early performance is more important:**

- **Financial trading**: Early profits compound over time
- **Real-time systems**: Immediate response matters most  
- **Resource allocation**: Early efficiency gains compound
- **Risk management**: Future uncertainty increases
- **Online learning**: Adapt quickly to current conditions

**Example**: Emergency response system where immediate action saves lives.

### Backward Discounting (Reverse)
**Use when later performance is more important:**

- **Final exam preparation**: End performance matters most
- **System warm-up**: Later performance more representative
- **Long-term optimization**: Steady-state behavior important
- **Quality improvement**: Final output quality critical
- **Convergence problems**: Solution quality at end matters

**Example**: Manufacturing process where final product quality is what counts.

## Mathematical Impact

With `discount_factor=0.9` over 100 timesteps:

### Forward Discounting
- **First 50 timesteps**: 99.4% of total weight
- **Last 50 timesteps**: 0.6% of total weight
- **Focus**: Immediate performance

### Backward Discounting  
- **First 50 timesteps**: 0.6% of total weight
- **Last 50 timesteps**: 99.4% of total weight
- **Focus**: End-game performance

## Implementation Details

### Fitness Function Changes
The `default_fitness_function` now includes discount direction logic:

```python
if discount_direction == 'forward':
    # Standard discounting: earlier rewards worth more
    discounted_reward = reward * (discount_factor ** t)
elif discount_direction == 'backward':
    # Reverse discounting: later rewards worth more
    reverse_t = n_timesteps - 1 - t
    discounted_reward = reward * (discount_factor ** reverse_t)
```

### String Representation
The model's string representation shows the discount direction:

```python
# Forward (default)
"GATreeActionSelector(actions=[0, 1, 2], discount=0.9, depth=5, size=17)"

# Backward  
"GATreeActionSelector(actions=[0, 1, 2], discount=0.9 (backward), depth=5, size=17)"
```

## Example Comparison

### Scenario: Resource Allocation Over Time

```python
# Generate time series with different reward patterns
# Early period: high demand, low costs
# Late period: moderate demand, high costs

# Forward discounting optimizes for early period
forward_selector = GATreeActionSelector(
    action_space=[0, 1, 2, 3, 4, 5],
    reward_function=resource_reward,
    discount_direction='forward'
)

# Backward discounting optimizes for late period  
backward_selector = GATreeActionSelector(
    action_space=[0, 1, 2, 3, 4, 5],
    reward_function=resource_reward,
    discount_direction='backward'
)

# Results:
# Forward: Aggressive early allocation, conservative later
# Backward: Conservative early allocation, optimized later
```

## Demo Script

Run the comparison demo to see the difference:

```bash
poetry run python examples/discount_comparison_demo.py
```

This script shows:
- Discount weight visualization
- Action sequence differences
- Performance comparison
- Use case recommendations

## Testing

The feature includes comprehensive tests:

```python
# Test backward discounting initialization
def test_backward_discounting_initialization(self):
    selector = GATreeActionSelector(
        action_space=[0, 1, 2],
        reward_function=reward_func,
        discount_direction='backward'
    )
    assert selector.discount_direction == 'backward'

# Test invalid direction
def test_invalid_discount_direction(self):
    with pytest.raises(ValueError):
        GATreeActionSelector(
            action_space=[0, 1, 2],
            reward_function=reward_func,
            discount_direction='invalid'
        )
```

## Backward Compatibility

This feature is **fully backward compatible**:
- Default behavior unchanged (`discount_direction='forward'`)
- Existing code continues to work without modification
- New parameter is optional

## Performance Considerations

- **Training time**: Same as forward discounting
- **Memory usage**: No additional memory overhead
- **Computational cost**: Minimal additional cost for reverse timestep calculation

## Real-World Applications

### Manufacturing Quality Control
```python
# Optimize for final product quality
quality_selector = GATreeActionSelector(
    action_space=['low_temp', 'med_temp', 'high_temp'],
    reward_function=quality_reward,
    discount_direction='backward',  # Final quality matters most
    discount_factor=0.95
)
```

### Academic Performance
```python
# Optimize for final exam performance
study_selector = GATreeActionSelector(
    action_space=['review', 'practice', 'rest'],
    reward_function=learning_reward,
    discount_direction='backward',  # Final performance matters
    discount_factor=0.9
)
```

### System Optimization
```python
# Optimize for steady-state performance
system_selector = GATreeActionSelector(
    action_space=['conservative', 'moderate', 'aggressive'],
    reward_function=system_reward,
    discount_direction='backward',  # Steady-state matters
    discount_factor=0.92
)
```

## Summary

Backward discounting provides a powerful new capability for scenarios where:
1. **End performance is more critical than early performance**
2. **System reaches steady-state and later behavior is more representative**
3. **Final outcomes determine overall success**

This feature expands GATreeActionSelector's applicability to a broader range of sequential decision problems while maintaining full backward compatibility.