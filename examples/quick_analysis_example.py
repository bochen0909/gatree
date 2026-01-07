#!/usr/bin/env python3
"""
Quick GATree Analysis Example

A simple example showing how to use the new analysis features.
"""

import json
import pandas as pd
from sklearn.datasets import load_iris
from gatree.methods.gatreeclassifier import GATreeClassifier

# Load data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

# Train model
print("Training GATree...")
clf = GATreeClassifier(max_depth=3, random_state=42)
clf.fit(X, y, population_size=30, max_iter=50)

# 1. Export tree to JSON
print("\n1. Exporting tree to JSON...")
tree_json = clf.export_tree_json()
with open('simple_tree.json', 'w') as f:
    json.dump(tree_json, f, indent=2)
print("✓ Tree exported to simple_tree.json")

# 2. Get feature importance
print("\n2. Feature importance:")
importance = clf.get_feature_importance()
for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
    print(f"   {feature}: {score:.3f}")

# 3. SHAP analysis
print("\n3. SHAP analysis:")
try:
    import shap
    sklearn_tree = clf.to_sklearn_tree()
    explainer = shap.TreeExplainer(sklearn_tree)
    shap_values = explainer.shap_values(X.iloc[:3])
    print("✓ SHAP values calculated successfully")
    print(f"   Shape: {shap_values[0].shape if isinstance(shap_values, list) else shap_values.shape}")
except ImportError:
    print("   SHAP not available (install with: pip install shap)")

print("\n✓ Analysis complete!")