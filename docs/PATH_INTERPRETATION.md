# Path Interpretation in GATree

GATree now supports decision path interpretation, allowing you to understand how the evolved decision trees make their predictions. This feature is available for all GATree classes: `GATreeClassifier`, `GATreeRegressor`, `GATreeActionSelector`, and `GATreeContinuousActionSelector`.

## Overview

The path interpretation functionality provides:
- **Decision paths**: Step-by-step breakdown of how a prediction is made
- **Feature importance**: Understanding which features are used most frequently
- **Decision patterns**: Analysis of common decision patterns across predictions
- **Interpretability**: Human-readable explanations of tree decisions

## New Methods

### For Classification and Regression

#### `predict_with_path(X)`
Returns predictions along with decision paths for multiple instances.

```python
from gatree.methods import GATreeClassifier
import pandas as pd

# Train your model
classifier = GATreeClassifier(max_depth=3)
classifier.fit(X_train, y_train)

# Get predictions with paths
predictions, paths = classifier.predict_with_path(X_test)
```

#### `predict_single_with_path(X_instance)`
Returns prediction and decision path for a single instance.

```python
# Single prediction with path
prediction, path = classifier.predict_single_with_path(X_test.iloc[0])
```

### For Action Selection

#### `predict_actions_with_path(X)`
Returns action sequence along with decision paths for time series data.

```python
from gatree.methods import GATreeActionSelector

# Train your action selector
selector = GATreeActionSelector(action_space=['buy', 'sell', 'hold'], 
                               reward_function=my_reward_function)
selector.fit(X_train, Y_train)

# Get actions with paths
actions, paths = selector.predict_actions_with_path(X_test)
```

#### `predict_single_action_with_path(X_instance)`
Returns action and decision path for a single timestep.

```python
# Single action with path
action, path = selector.predict_single_action_with_path(X_test.iloc[0])
```

### For Continuous Action Selection

#### `predict_actions_with_path(X)`
Returns continuous action sequence along with decision paths.

```python
from gatree.methods import GATreeContinuousActionSelector

# Train your continuous action selector
selector = GATreeContinuousActionSelector(reward_function=my_reward_function,
                                         action_bounds=(0, 1))
selector.fit(X_train, Y_train)

# Get continuous actions with paths
actions, paths = selector.predict_actions_with_path(X_test)
```

## Decision Path Structure

Each decision path is a list of dictionaries representing the steps taken through the tree:

### Internal Node Step
```python
{
    'node_type': 'internal',
    'feature_index': 2,
    'feature_name': 'feature_2',
    'feature_value': 1.5,
    'threshold': 1.0,
    'condition': 'feature_2 > 1.0',
    'decision': True  # True if condition is met, False otherwise
}
```

### Leaf Node Step
```python
{
    'node_type': 'leaf',
    'predicted_value': 1,
    'leaf_value': 1
}
```

### Error Handling
```python
{
    'node_type': 'error',
    'error_message': 'Error description',
    'default_value': 0
}
```

## Utility Functions

### Formatting Decision Paths

```python
from gatree.utils import format_decision_path, print_decision_path

# Format a path as a string
formatted_path = format_decision_path(path, feature_names=['temp', 'humidity', 'pressure'])

# Print a formatted path
print_decision_path(path, feature_names=['temp', 'humidity', 'pressure'], 
                   title="My Decision Path")
```

### Analyzing Multiple Paths

```python
from gatree.utils import analyze_decision_paths, print_path_analysis

# Analyze patterns across multiple paths
analysis = analyze_decision_paths(paths, feature_names=['temp', 'humidity', 'pressure'])

# Print the analysis
print_path_analysis(analysis)
```

The analysis includes:
- **Feature usage frequency**: How often each feature is used in decisions
- **Path length statistics**: Average, min, and max path lengths
- **Common decision patterns**: Most frequent decision combinations
- **Prediction distribution**: Distribution of leaf values

## Complete Example

```python
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from gatree.methods import GATreeClassifier
from gatree.utils import print_decision_path, analyze_decision_paths, print_path_analysis

# Generate sample data
X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
X_df = pd.DataFrame(X, columns=['temperature', 'humidity', 'pressure', 'wind_speed'])
y_series = pd.Series(y)

# Train classifier
classifier = GATreeClassifier(max_depth=4, random_state=42)
classifier.fit(X_df, y_series, population_size=50, max_iter=20)

# Make predictions with paths
predictions, paths = classifier.predict_with_path(X_df.head(5))

# Show individual decision paths
feature_names = X_df.columns.tolist()
for i, (pred, path) in enumerate(zip(predictions, paths)):
    print_decision_path(path, feature_names, f"Sample {i+1} (Predicted: {pred})")

# Analyze all paths
all_predictions, all_paths = classifier.predict_with_path(X_df)
analysis = analyze_decision_paths(all_paths, feature_names)
print_path_analysis(analysis)

# Single prediction example
single_prediction, single_path = classifier.predict_single_with_path(X_df.iloc[0])
print(f"Single prediction: {single_prediction}")
print_decision_path(single_path, feature_names, "Single Instance Path")
```

## Benefits

1. **Model Interpretability**: Understand exactly how predictions are made
2. **Feature Importance**: Identify which features are most influential
3. **Debugging**: Detect potential issues in model behavior
4. **Trust**: Build confidence in model decisions through transparency
5. **Compliance**: Meet regulatory requirements for explainable AI

## Use Cases

- **Medical Diagnosis**: Explain why a particular diagnosis was made
- **Financial Trading**: Understand trading decisions in action selection
- **Quality Control**: Interpret classification decisions in manufacturing
- **Risk Assessment**: Explain risk predictions with clear reasoning
- **Scientific Research**: Understand patterns discovered by the model

## Performance Considerations

- Path interpretation adds minimal computational overhead
- Paths are generated during prediction, not stored permanently
- Memory usage scales with the number of predictions and tree depth
- For large datasets, consider analyzing paths in batches

## Integration with Existing Code

The path interpretation functionality is fully backward compatible. Existing code will continue to work unchanged, and you can add path interpretation incrementally where needed.

```python
# Existing code continues to work
predictions = classifier.predict(X_test)

# Add path interpretation when needed
predictions_with_paths, paths = classifier.predict_with_path(X_test)
```