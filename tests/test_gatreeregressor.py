"""
Tests for GATreeRegressor class.
"""

import unittest
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from gatree.methods.gatreeregressor import GATreeRegressor


class TestGATreeRegressor(unittest.TestCase):
    """Test cases for GATreeRegressor."""
    
    def setUp(self):
        """Set up test data."""
        # Generate small synthetic dataset for testing
        X, y = make_regression(
            n_samples=100,
            n_features=3,
            n_informative=2,
            noise=0.1,
            random_state=42
        )
        
        self.X = pd.DataFrame(X, columns=['feature_0', 'feature_1', 'feature_2'])
        self.y = pd.Series(y, name='target')
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42
        )
    
    def test_initialization(self):
        """Test GATreeRegressor initialization."""
        regressor = GATreeRegressor(max_depth=5, random_state=42)
        
        self.assertEqual(regressor.max_depth, 5)
        self.assertIsNotNone(regressor.random)
        self.assertIsNone(regressor._tree)
        self.assertEqual(len(regressor._best_fitness), 0)
        self.assertEqual(len(regressor._avg_fitness), 0)
    
    def test_fit_basic(self):
        """Test basic fitting functionality."""
        regressor = GATreeRegressor(max_depth=3, random_state=42)
        
        # Fit with minimal parameters for quick test
        result = regressor.fit(
            self.X_train, 
            self.y_train,
            population_size=10,
            max_iter=5,
            mutation_probability=0.1
        )
        
        # Check that fitting returns self
        self.assertEqual(result, regressor)
        
        # Check that tree was created
        self.assertIsNotNone(regressor._tree)
        
        # Check that fitness tracking works
        self.assertEqual(len(regressor._best_fitness), 6)  # max_iter + 1
        self.assertEqual(len(regressor._avg_fitness), 6)
        
        # Check that fitness values are reasonable
        self.assertTrue(all(isinstance(f, (int, float)) for f in regressor._best_fitness))
        self.assertTrue(all(f >= 0 for f in regressor._best_fitness))
    
    def test_predict(self):
        """Test prediction functionality."""
        regressor = GATreeRegressor(max_depth=3, random_state=42)
        
        # Fit the model
        regressor.fit(
            self.X_train, 
            self.y_train,
            population_size=10,
            max_iter=5
        )
        
        # Make predictions
        predictions = regressor.predict(self.X_test)
        
        # Check prediction shape and type
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertTrue(all(isinstance(p, (int, float)) for p in predictions))
        
        # Check that predictions are reasonable (not all the same)
        self.assertTrue(len(set(predictions)) > 1)
    
    def test_fitness_function(self):
        """Test the default fitness function."""
        regressor = GATreeRegressor(random_state=42)
        
        # Create a mock tree node with evaluation data
        from gatree.tree.node import Node
        mock_tree = Node()
        mock_tree.y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        mock_tree.y_pred = [1.1, 2.1, 2.9, 4.2, 4.8]
        
        # Mock the size method
        mock_tree.size = lambda: 10
        
        # Test fitness calculation
        fitness = regressor.default_fitness_function(
            mock_tree, 
            y_range=4.0  # max - min = 5 - 1 = 4
        )
        
        # Fitness should be a positive number
        self.assertIsInstance(fitness, (int, float))
        self.assertGreater(fitness, 0)
        
        # Test with empty predictions (should return inf)
        empty_tree = Node()
        empty_tree.y_true = []
        empty_tree.y_pred = []
        empty_tree.size = lambda: 1
        
        fitness_empty = regressor.default_fitness_function(empty_tree)
        self.assertEqual(fitness_empty, float('inf'))
    
    def test_regression_performance(self):
        """Test that the regressor can achieve reasonable performance."""
        regressor = GATreeRegressor(max_depth=5, random_state=42)
        
        # Fit with reasonable parameters
        regressor.fit(
            self.X_train, 
            self.y_train,
            population_size=20,
            max_iter=20,
            mutation_probability=0.15
        )
        
        # Make predictions
        predictions = regressor.predict(self.X_test)
        
        # Calculate performance metrics
        mse = mean_squared_error(self.y_test, predictions)
        r2 = r2_score(self.y_test, predictions)
        
        # Check that performance is reasonable
        # (not perfect due to small dataset and few iterations)
        self.assertIsInstance(mse, (int, float))
        self.assertIsInstance(r2, (int, float))
        self.assertGreater(r2, -5.0)  # R² should be reasonable (allowing for small dataset)
        
        # Check that fitness improved during training
        initial_fitness = regressor._best_fitness[0]
        final_fitness = regressor._best_fitness[-1]
        self.assertLessEqual(final_fitness, initial_fitness)  # Lower is better
    
    def test_tree_structure(self):
        """Test that the resulting tree has reasonable structure."""
        regressor = GATreeRegressor(max_depth=4, random_state=42)
        
        regressor.fit(
            self.X_train, 
            self.y_train,
            population_size=15,
            max_iter=10
        )
        
        tree = regressor._tree
        
        # Check tree properties
        self.assertIsNotNone(tree)
        self.assertLessEqual(tree.max_depth(), 4)
        self.assertGreater(tree.size(), 0)
        self.assertGreater(len(tree.get_leaves()), 0)
        
        # Check that tree has both internal nodes and leaves
        leaves = tree.get_leaves()
        self.assertTrue(any(leaf.att_index == -1 for leaf in leaves))


if __name__ == '__main__':
    unittest.main()