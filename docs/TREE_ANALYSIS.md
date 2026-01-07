# GATree Analysis Features

This document describes the new analysis capabilities added to GATree for tree visualization, feature importance, and explainability.

## Overview

GATree now supports three major analysis features:

1. **JSON Export**: Export tree structure to JSON format for visualization and analysis
2. **Feature Importance**: Calculate feature importance scores based on tree structure
3. **SHAP Integration**: Convert trees to sklearn format for SHAP explainability analysis

## Features

### 1. JSON Export

Export your trained GATree to a structured JSON format that can be used for visualization, analysis, or integration with other tools.

```python
from gatree.methods.gatreeclassifier import GATreeClassifier
import json

# Train your model
clf = GATreeClassifier(max_depth=4)
clf.fit(X_train, y_train)

# Export to JSON
tree_json = clf.export_tree_json()

# Save to file
with open('tree_structure.json', 'w') as f:
    json.dump(tree_json, f, indent=2)
```

**JSON Structure:**
```json
{
  "node_id": 12345,
  "is_leaf": false,
  "feature": "sepal_length",
  "feature_index": 0,
  "threshold": 5.45,
  "samples": 120,
  "left": {
    "node_id": 12346,
    "is_leaf": true,
    "value": 0,
    "samples": 50
  },
  "right": { ... }
}
```

**Parameters:**
- `node` (optional): Specific node to export. Defaults to root.
- `feature_names` (optional): Custom feature names. Defaults to training data column names.

### 2. Feature Importance

Calculate feature importance based on how frequently and effectively each feature is used in the tree structure.

```python
# Get feature importance scores
importance = clf.get_feature_importance()

# Display results
for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
    print(f"{feature}: {score:.4f}")
```

**How it works:**
- Traverses the entire tree structure
- Weights features by their depth (closer to root = higher importance)
- Weights features by number of samples passing through each node
- Normalizes scores to sum to 1.0

**Returns:**
Dictionary with feature names as keys and importance scores (0.0 to 1.0) as values.

### 3. SHAP Integration

Convert GATree models to sklearn format for compatibility with SHAP (SHapley Additive exPlanations) explainability tools.

```python
import shap

# Convert GATree to sklearn format
sklearn_tree = clf.to_sklearn_tree()

# Create SHAP explainer
explainer = shap.TreeExplainer(sklearn_tree)

# Calculate SHAP values
shap_values = explainer.shap_values(X_test)

# Visualize (requires matplotlib)
shap.summary_plot(shap_values, X_test)
```

**Parameters:**
- `tree_type` (optional): 'classifier', 'regressor', or 'auto' (default)

**Note:** The sklearn conversion creates a tree fitted on the same training data. While it may not have identical structure to the GATree, it provides SHAP compatibility and similar predictive behavior.

## Installation

To use the SHAP integration feature, install SHAP:

```bash
pip install shap
# or
poetry add shap
```

SHAP is now included as a dependency in the updated `pyproject.toml`.

## Examples

### Complete Analysis Workflow

```python
import pandas as pd
from sklearn.datasets import load_iris
from gatree.methods.gatreeclassifier import GATreeClassifier
import json
import shap

# Load data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

# Train model
clf = GATreeClassifier(max_depth=4, random_state=42)
clf.fit(X, y, population_size=100, max_iter=200)

# 1. Visualize tree structure
print("Tree Structure:")
clf.plot()

# 2. Export to JSON
tree_json = clf.export_tree_json()
with open('iris_tree.json', 'w') as f:
    json.dump(tree_json, f, indent=2)
print("Tree exported to iris_tree.json")

# 3. Feature importance
importance = clf.get_feature_importance()
print("\nFeature Importance:")
for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
    print(f"  {feature}: {score:.4f}")

# 4. SHAP analysis
sklearn_tree = clf.to_sklearn_tree()
explainer = shap.TreeExplainer(sklearn_tree)
shap_values = explainer.shap_values(X)

print(f"\nSHAP values calculated for {len(X)} samples")
print(f"SHAP values shape: {shap_values[0].shape if isinstance(shap_values, list) else shap_values.shape}")
```

### Regression Example

```python
from gatree.methods.gatreeregressor import GATreeRegressor
from sklearn.datasets import make_regression

# Create regression data
X, y = make_regression(n_samples=200, n_features=4, noise=0.1, random_state=42)
X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(4)])
y = pd.Series(y)

# Train regressor
reg = GATreeRegressor(max_depth=5, random_state=42)
reg.fit(X, y, population_size=100, max_iter=150)

# Analyze
importance = reg.get_feature_importance()
sklearn_tree = reg.to_sklearn_tree()
explainer = shap.TreeExplainer(sklearn_tree)
shap_values = explainer.shap_values(X[:10])  # First 10 samples

print("Top features by importance:")
for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]:
    print(f"  {feature}: {score:.4f}")
```

## API Reference

### GATree.export_tree_json(node=None, feature_names=None)

Export tree structure to JSON format.

**Parameters:**
- `node` (Node, optional): Node to export. Defaults to fitted tree root.
- `feature_names` (list, optional): Feature names. Defaults to training data columns.

**Returns:**
- `dict`: JSON representation of tree structure.

### GATree.get_feature_importance()

Calculate feature importance scores.

**Returns:**
- `dict`: Feature names mapped to importance scores (0.0 to 1.0).

### GATree.to_sklearn_tree(tree_type='auto')

Convert to sklearn DecisionTree format.

**Parameters:**
- `tree_type` (str): 'classifier', 'regressor', or 'auto'.

**Returns:**
- `sklearn.tree.DecisionTreeClassifier/Regressor`: Fitted sklearn tree.

## Limitations and Notes

1. **SHAP Conversion**: The sklearn conversion creates a tree fitted on the same data rather than replicating the exact GATree structure. This provides SHAP compatibility while maintaining similar predictive behavior.

2. **Feature Importance**: The importance calculation is based on tree structure analysis, not traditional metrics like Gini importance. It reflects how the genetic algorithm utilized features during evolution.

3. **JSON Export**: Large trees may produce large JSON files. Consider limiting tree depth for visualization purposes.

4. **Memory Usage**: SHAP analysis can be memory-intensive for large datasets. Consider using subsets for initial analysis.

## Testing

Run the test suite to verify functionality:

```bash
python -m pytest tests/test_tree_analysis.py -v
```

Or run the comprehensive demo:

```bash
python examples/tree_analysis_demo.py
```

## Integration with Existing Code

These new methods are backward-compatible additions to GATree. Existing code will continue to work unchanged, and you can gradually adopt the new analysis features as needed.

The methods are available on all GATree variants:
- `GATreeClassifier`
- `GATreeRegressor` 
- `GATreeActionSelector`

## Future Enhancements

Potential future improvements:
1. Direct graphical tree visualization (e.g., with graphviz)
2. Interactive tree exploration tools
3. More sophisticated feature importance metrics
4. Direct SHAP explainer implementation for GATree
5. Tree comparison and ensemble analysis tools