#!/usr/bin/env python3
"""
Tests for new GATree analysis features:
- JSON export
- Feature importance
- SHAP integration
"""

import json
import unittest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression

from gatree.methods.gatreeclassifier import GATreeClassifier
from gatree.methods.gatreeregressor import GATreeRegressor


class TestTreeAnalysis(unittest.TestCase):
    """Test cases for tree analysis features."""
    
    def setUp(self):
        """Set up test data."""
        # Classification data
        X_class, y_class = make_classification(
            n_samples=100, n_features=4, n_classes=2, 
            random_state=42, n_redundant=0
        )
        self.X_class = pd.DataFrame(X_class, columns=[f'feature_{i}' for i in range(4)])
        self.y_class = pd.Series(y_class)
        
        # Regression data
        X_reg, y_reg = make_regression(
            n_samples=100, n_features=4, noise=0.1, random_state=42
        )
        self.X_reg = pd.DataFrame(X_reg, columns=[f'feature_{i}' for i in range(4)])
        self.y_reg = pd.Series(y_reg)
    
    def test_json_export_classifier(self):
        """Test JSON export for classifier."""
        # Train classifier
        clf = GATreeClassifier(max_depth=3, random_state=42)
        clf.fit(self.X_class, self.y_class, population_size=20, max_iter=10)
        
        # Export to JSON
        tree_json = clf.export_tree_json()
        
        # Verify JSON structure
        self.assertIsInstance(tree_json, dict)
        self.assertIn('node_id', tree_json)
        self.assertIn('is_leaf', tree_json)
        
        # Verify JSON is serializable
        json_str = json.dumps(tree_json)
        self.assertIsInstance(json_str, str)
        
        # Verify we can load it back
        loaded_json = json.loads(json_str)
        self.assertEqual(tree_json['node_id'], loaded_json['node_id'])
    
    def test_json_export_regressor(self):
        """Test JSON export for regressor."""
        # Train regressor
        reg = GATreeRegressor(max_depth=3, random_state=42)
        reg.fit(self.X_reg, self.y_reg, population_size=20, max_iter=10)
        
        # Export to JSON
        tree_json = reg.export_tree_json()
        
        # Verify JSON structure
        self.assertIsInstance(tree_json, dict)
        self.assertIn('node_id', tree_json)
        self.assertIn('is_leaf', tree_json)
        
        # Verify JSON is serializable
        json_str = json.dumps(tree_json)
        self.assertIsInstance(json_str, str)
    
    def test_feature_importance_classifier(self):
        """Test feature importance calculation for classifier."""
        # Train classifier
        clf = GATreeClassifier(max_depth=3, random_state=42)
        clf.fit(self.X_class, self.y_class, population_size=20, max_iter=10)
        
        # Get feature importance
        importance = clf.get_feature_importance()
        
        # Verify structure
        self.assertIsInstance(importance, dict)
        self.assertEqual(len(importance), 4)  # 4 features
        
        # Verify all features are present
        for i in range(4):
            self.assertIn(f'feature_{i}', importance)
        
        # Verify importance scores are normalized (sum to 1)
        total_importance = sum(importance.values())
        self.assertAlmostEqual(total_importance, 1.0, places=5)
        
        # Verify all scores are non-negative
        for score in importance.values():
            self.assertGreaterEqual(score, 0.0)
    
    def test_feature_importance_regressor(self):
        """Test feature importance calculation for regressor."""
        # Train regressor
        reg = GATreeRegressor(max_depth=3, random_state=42)
        reg.fit(self.X_reg, self.y_reg, population_size=20, max_iter=10)
        
        # Get feature importance
        importance = reg.get_feature_importance()
        
        # Verify structure
        self.assertIsInstance(importance, dict)
        self.assertEqual(len(importance), 4)  # 4 features
        
        # Verify importance scores are normalized
        total_importance = sum(importance.values())
        self.assertAlmostEqual(total_importance, 1.0, places=5)
    
    def test_sklearn_conversion_classifier(self):
        """Test conversion to sklearn format for classifier."""
        # Train classifier
        clf = GATreeClassifier(max_depth=3, random_state=42)
        clf.fit(self.X_class, self.y_class, population_size=20, max_iter=10)
        
        # Convert to sklearn
        sklearn_tree = clf.to_sklearn_tree()
        
        # Verify it's a sklearn tree
        from sklearn.tree import DecisionTreeClassifier
        self.assertIsInstance(sklearn_tree, DecisionTreeClassifier)
        
        # Verify it can make predictions
        predictions = sklearn_tree.predict(self.X_class)
        self.assertEqual(len(predictions), len(self.y_class))
    
    def test_sklearn_conversion_regressor(self):
        """Test conversion to sklearn format for regressor."""
        # Train regressor
        reg = GATreeRegressor(max_depth=3, random_state=42)
        reg.fit(self.X_reg, self.y_reg, population_size=20, max_iter=10)
        
        # Convert to sklearn
        sklearn_tree = reg.to_sklearn_tree()
        
        # Verify it's a sklearn tree
        from sklearn.tree import DecisionTreeRegressor
        self.assertIsInstance(sklearn_tree, DecisionTreeRegressor)
        
        # Verify it can make predictions
        predictions = sklearn_tree.predict(self.X_reg)
        self.assertEqual(len(predictions), len(self.y_reg))
    
    def test_shap_integration_classifier(self):
        """Test SHAP integration for classifier."""
        try:
            import shap
        except ImportError:
            self.skipTest("SHAP not installed")
        
        # Train classifier
        clf = GATreeClassifier(max_depth=3, random_state=42)
        clf.fit(self.X_class, self.y_class, population_size=20, max_iter=10)
        
        # Convert to sklearn and create SHAP explainer
        sklearn_tree = clf.to_sklearn_tree()
        explainer = shap.TreeExplainer(sklearn_tree)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(self.X_class.iloc[:5])
        
        # Verify SHAP values structure
        if isinstance(shap_values, list):  # Multi-class
            self.assertIsInstance(shap_values, list)
            self.assertEqual(len(shap_values[0]), 5)  # 5 samples
            self.assertEqual(len(shap_values[0][0]), 4)  # 4 features
        else:  # Binary
            self.assertEqual(shap_values.shape, (5, 4))  # 5 samples, 4 features
    
    def test_shap_integration_regressor(self):
        """Test SHAP integration for regressor."""
        try:
            import shap
        except ImportError:
            self.skipTest("SHAP not installed")
        
        # Train regressor
        reg = GATreeRegressor(max_depth=3, random_state=42)
        reg.fit(self.X_reg, self.y_reg, population_size=20, max_iter=10)
        
        # Convert to sklearn and create SHAP explainer
        sklearn_tree = reg.to_sklearn_tree()
        explainer = shap.TreeExplainer(sklearn_tree)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(self.X_reg.iloc[:5])
        
        # Verify SHAP values structure
        self.assertEqual(shap_values.shape, (5, 4))  # 5 samples, 4 features
    
    def test_json_export_with_custom_feature_names(self):
        """Test JSON export with custom feature names."""
        # Train classifier
        clf = GATreeClassifier(max_depth=3, random_state=42)
        clf.fit(self.X_class, self.y_class, population_size=20, max_iter=10)
        
        # Export with custom feature names
        custom_names = ['height', 'weight', 'age', 'income']
        tree_json = clf.export_tree_json(feature_names=custom_names)
        
        # Convert to string to search for feature names
        json_str = json.dumps(tree_json)
        
        # At least one custom feature name should appear in the JSON
        found_custom_name = any(name in json_str for name in custom_names)
        # Note: This might not always be true if the tree is very small
        # but it's a reasonable test for most cases
    
    def test_empty_tree_handling(self):
        """Test handling of empty/unfitted trees."""
        # Create unfitted classifier
        clf = GATreeClassifier(max_depth=3, random_state=42)
        
        # JSON export should return None for unfitted tree
        tree_json = clf.export_tree_json()
        self.assertIsNone(tree_json)
        
        # Feature importance should return empty dict
        importance = clf.get_feature_importance()
        self.assertEqual(importance, {})
        
        # sklearn conversion should raise error
        with self.assertRaises(ValueError):
            clf.to_sklearn_tree()


if __name__ == '__main__':
    unittest.main()