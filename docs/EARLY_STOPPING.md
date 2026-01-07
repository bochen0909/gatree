# Early Stopping in GATree

This document describes the early stopping functionality implemented in GATree methods to prevent overfitting and improve training efficiency.

## Overview

Early stopping is a regularization technique that monitors the fitness/loss during training and stops the evolutionary process when no improvement is observed for a specified number of generations. This helps prevent overfitting and saves computational resources.

## Supported Methods

Early stopping is available in all GATree methods:
- `GATreeClassifier`
- `GATreeRegressor` 
- `GATreeActionSelector`

## Parameters

All GATree methods now support the following early stopping parameters in their `fit()` method:

### `early_stopping` (bool, default=False)
- Enables or disables early stopping
- When `False`, training runs for the full `max_iter` generations
- When `True`, training may stop early based on other parameters

### `patience` (int, default=50)
- Number of generations to wait for improvement before stopping
- Training stops if no improvement is seen for `patience` consecutive generations
- Must be positive when `early_stopping=True`

### `min_delta` (float, default=1e-4)
- Minimum change in fitness to qualify as an improvement
- A fitness change smaller than `min_delta` is not considered an improvement
- Must be non-negative

### `restore_best_weights` (bool, default=True)
- Whether to restore the best tree when early stopping occurs
- When `True`, the final model is the best tree found during training
- When `False`, the final model is the tree from the last generation

## Usage Examples

### Basic Early Stopping

```python
from gatree.methods.gatreeclassifier import GATreeClassifier

classifier = GATreeClassifier(random_state=42)
classifier.fit(
    X, y,
    population_size=100,
    max_iter=1000,
    early_stopping=True,
    patience=50,
    min_delta=0.001
)
```

### With Progress Monitoring

```python
def progress_callback(generation, best_fitness, avg_fitness):
    print(f"Generation {generation}: Best={best_fitness:.4f}")

classifier.fit(
    X, y,
    early_stopping=True,
    patience=20,
    min_delta=0.01,
    progress_callback=progress_callback
)
```

### GATreeRegressor Example

```python
from gatree.methods.gatreeregressor import GATreeRegressor

regressor = GATreeRegressor(random_state=42)
regressor.fit(
    X, y,
    population_size=150,
    max_iter=500,
    early_stopping=True,
    patience=30,
    min_delta=0.005,
    restore_best_weights=True
)
```

### GATreeActionSelector Example

```python
from gatree.methods.gatreeactionselector import GATreeActionSelector

action_selector = GATreeActionSelector(
    action_space=['buy', 'sell', 'hold'],
    reward_function=my_reward_function
)

action_selector.fit(
    X, Y,
    population_size=100,
    max_iter=800,
    early_stopping=True,
    patience=40,
    min_delta=0.1
)
```

## How It Works

1. **Fitness Tracking**: During training, the best fitness value is tracked across generations
2. **Improvement Detection**: After each generation, the algorithm checks if the current best fitness is better than the previous best by at least `min_delta`
3. **Patience Counter**: If no improvement is detected, a patience counter is incremented
4. **Early Stopping**: When the patience counter reaches the `patience` threshold, training stops
5. **Best Model Restoration**: If `restore_best_weights=True`, the best tree found during training is restored as the final model

## Benefits

### Computational Efficiency
- Reduces training time by stopping when no further improvement is expected
- Saves computational resources, especially important for large populations or long training runs

### Overfitting Prevention
- Prevents the model from continuing to evolve when it's no longer improving on the fitness function
- Helps maintain generalization performance

### Automatic Hyperparameter Tuning
- Reduces the need to manually tune `max_iter` for different datasets
- Allows setting a high `max_iter` with confidence that training will stop at the right time

## Best Practices

### Choosing Patience
- **Small datasets**: Use smaller patience values (10-30 generations)
- **Large datasets**: Use larger patience values (50-100 generations)
- **Complex problems**: May require higher patience to allow for gradual improvement

### Setting min_delta
- **Classification**: Typically 0.001-0.01 depending on the scale of your fitness function
- **Regression**: Adjust based on the scale of your target variable and fitness function
- **Action Selection**: May need larger values (0.1-1.0) due to the nature of reward functions

### Population Size Considerations
- Larger populations may need higher patience values as they can take longer to converge
- Smaller populations may benefit from lower patience values

## Monitoring Training

Use progress callbacks to monitor the early stopping process:

```python
def detailed_callback(generation, best_fitness, avg_fitness, best_reward=None):
    if best_reward is not None:  # GATreeActionSelector
        print(f"Gen {generation:3d}: Fitness={best_fitness:.4f}, "
              f"Avg={avg_fitness:.4f}, Reward={best_reward:.4f}")
    else:  # GATreeClassifier/Regressor
        print(f"Gen {generation:3d}: Fitness={best_fitness:.4f}, "
              f"Avg={avg_fitness:.4f}")

# The callback will show when early stopping occurs
classifier.fit(X, y, early_stopping=True, progress_callback=detailed_callback)
```

## Validation

After training with early stopping, you can check:

```python
# Number of generations actually run
print(f"Generations run: {len(classifier._best_fitness)}")

# Best fitness achieved
print(f"Best fitness: {min(classifier._best_fitness):.4f}")

# Fitness evolution
import matplotlib.pyplot as plt
plt.plot(classifier._best_fitness, label='Best Fitness')
plt.plot(classifier._avg_fitness, label='Average Fitness')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend()
plt.show()
```

## Troubleshooting

### Early Stopping Too Aggressive
- Increase `patience` value
- Decrease `min_delta` value
- Check if your fitness function is noisy

### Early Stopping Not Triggering
- Decrease `patience` value
- Increase `min_delta` value
- Verify that your problem actually converges

### Performance Issues
- Early stopping adds minimal computational overhead
- The main cost is copying the best tree when `restore_best_weights=True`
- Consider setting `restore_best_weights=False` for very large trees if memory is a concern

## Implementation Details

The early stopping implementation:
- Tracks the best fitness value seen so far
- Compares current best fitness with historical best using `min_delta` threshold
- Maintains a patience counter that resets when improvement is detected
- Optionally stores a copy of the best tree for restoration
- Integrates seamlessly with existing progress callback functionality

This feature is backward compatible - existing code will work unchanged with `early_stopping=False` by default.