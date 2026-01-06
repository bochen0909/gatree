# GATree Regression Examples

This directory contains examples demonstrating how to use `GATreeRegressor` for regression tasks.

## Overview

The `GATreeRegressor` is an evolutionary decision tree regressor that uses genetic algorithms to evolve decision trees for continuous target prediction. It follows the same architectural patterns as `GATreeClassifier` and `GATreeClustering` but is optimized for regression tasks.

## Key Features

- **Evolutionary Optimization**: Uses genetic algorithms to evolve tree structures
- **Continuous Target Support**: Handles continuous target variables
- **MSE-based Fitness**: Uses normalized Mean Squared Error with complexity penalty
- **Parallel Processing**: Supports multi-core evaluation for faster training
- **Flexible Tree Structure**: Evolves both tree topology and split parameters

## Examples

### 1. Synthetic Regression (`synthetic_regression.py`)

A simple example using synthetic data generated with `make_regression`:

```python
from gatree.methods.gatreeregressor import GATreeRegressor

# Create regressor
gatree = GATreeRegressor(max_depth=8, n_jobs=2, random_state=42)

# Fit with evolutionary parameters
gatree.fit(
    X=X_train, 
    y=y_train, 
    population_size=30,
    max_iter=50,
    mutation_probability=0.2
)

# Make predictions
y_pred = gatree.predict(X_test)
```

### 2. California Housing (`boston_housing.py`)

A real-world example using the California Housing dataset:

```python
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

# Load and scale data
housing = fetch_california_housing()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train with larger parameters for better performance
gatree.fit(
    X=X_train_scaled, 
    y=y_train, 
    population_size=50,
    max_iter=100,
    mutation_probability=0.15,
    elite_size=2
)
```

## How It Works

### 1. **Feature Splitting**
- Uses the same binary threshold splitting as classification
- Splits are of the form: `if feature_i > threshold: go_left else: go_right`
- Thresholds are automatically generated as midpoints between unique feature values

### 2. **Leaf Values**
- Leaf nodes contain continuous target values
- Target range is discretized into bins for initial population generation
- Final predictions are the leaf values reached during tree traversal

### 3. **Fitness Function**
The default fitness function balances prediction accuracy with tree complexity:

```python
def default_fitness_function(root, **kwargs):
    mse = mean_squared_error(root.y_true, root.y_pred)
    y_range = kwargs.get('y_range', 1.0)
    normalized_mse = mse / (y_range ** 2)
    complexity_penalty = 0.002 * root.size()
    return normalized_mse + complexity_penalty
```

### 4. **Genetic Operations**
- **Selection**: Tournament selection picks best performers
- **Crossover**: Swaps subtrees between parent trees
- **Mutation**: Modifies tree structure (change features, thresholds, or topology)
- **Elitism**: Preserves best trees across generations

## Parameters

### Key Parameters for `fit()`:

- `population_size` (default: 150): Number of trees in each generation
- `max_iter` (default: 2000): Number of evolutionary generations
- `mutation_probability` (default: 0.1): Probability of mutating offspring
- `elite_size` (default: 1): Number of best trees preserved each generation
- `selection_tournament_size` (default: 2): Tournament size for parent selection

### Constructor Parameters:

- `max_depth`: Maximum tree depth (None for unlimited)
- `n_jobs`: Number of parallel processes for evaluation
- `random_state`: Seed for reproducibility
- `fitness_function`: Custom fitness function (optional)

## Performance Tips

1. **Feature Scaling**: Consider scaling features for better performance
2. **Population Size**: Larger populations explore more solutions but take longer
3. **Iterations**: More iterations generally improve results but increase training time
4. **Parallel Processing**: Use `n_jobs > 1` for faster evaluation on multi-core systems
5. **Tree Depth**: Limit `max_depth` to prevent overfitting

## Comparison with Traditional Methods

| Aspect | GATreeRegressor | Traditional Decision Trees |
|--------|-----------------|---------------------------|
| Split Selection | Evolutionary optimization | Greedy (MSE reduction) |
| Global Optimization | Yes (population-based) | No (local optima) |
| Parallelization | Built-in | Limited |
| Overfitting Control | Fitness-based complexity penalty | Pruning required |
| Training Time | Longer (evolutionary) | Faster (greedy) |
| Solution Quality | Potentially better global optimum | Local optimum |

## Running the Examples

```bash
# Run synthetic regression example
python examples/regression/synthetic_regression.py

# Run California housing example  
python examples/regression/boston_housing.py
```

## Expected Output

The examples will show:
- Dataset information
- Training progress
- Performance metrics (MSE, RMSE, R²)
- Sample predictions
- Tree structure information
- Fitness evolution (if matplotlib available)

## Requirements

- pandas
- numpy  
- scikit-learn
- joblib
- matplotlib (optional, for plotting)

## Notes

- The regressor works best with scaled features
- Performance depends heavily on population size and iterations
- Small datasets may show high variance in results
- Consider using cross-validation for robust performance evaluation