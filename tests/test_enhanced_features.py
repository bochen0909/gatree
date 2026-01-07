#!/usr/bin/env python3
"""
Tests for the enhanced GATree features:
1. Sample weight support in GATreeClassifier
2. Progress callbacks
3. Error handling improvements
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

from gatree.methods.gatreeclassifier import GATreeClassifier
from gatree.methods.gatreeregressor import GATreeRegressor
from gatree.methods.gatreeactionselector import GATreeActionSelector


class TestEnhancedFeatures(unittest.TestCase):
    
    def setUp(self):
        """Set up test data"""
        # Classification data
        X_cls, y_cls = make_classification(n_samples=50, n_features=5, n_classes=2, 
                                          n_informative=3, random_state=42)
        self.X_cls = pd.DataFrame(X_cls, columns=[f'feature_{i}' for i in range(X_cls.shape[1])])
        self.y_cls = pd.Series(y_cls)
        
        # Regression data
        X_reg, y_reg = make_regression(n_samples=50, n_features=5, noise=0.1, random_state=42)
        self.X_reg = pd.DataFrame(X_reg, columns=[f'feature_{i}' for i in range(X_reg.shape[1])])
        self.y_reg = pd.Series(y_reg)
        
        # Sample weights
        self.sample_weights = np.random.uniform(0.5, 2.0, size=50)
        
        # Progress callback tracking
        self.callback_calls = []
    
    def progress_callback(self, generation, best_fitness, avg_fitness, best_reward=None):
        """Test progress callback"""
        self.callback_calls.append({
            'generation': generation,
            'best_fitness': best_fitness,
            'avg_fitness': avg_fitness,
            'best_reward': best_reward
        })
    
    def test_classifier_sample_weights(self):
        """Test GATreeClassifier with sample weights"""
        classifier = GATreeClassifier(random_state=42)
        
        # Fit with sample weights
        classifier.fit(
            self.X_cls, self.y_cls,
            sample_weight=self.sample_weights,
            population_size=10,
            max_iter=3
        )
        
        # Check that classifier was fitted
        self.assertIsNotNone(classifier._tree)
        self.assertEqual(len(classifier._best_fitness), 4)  # max_iter + 1
        
        # Check predictions work
        predictions = classifier.predict(self.X_cls)
        self.assertEqual(len(predictions), len(self.y_cls))
    
    def test_classifier_progress_callback(self):
        """Test GATreeClassifier with progress callback"""
        classifier = GATreeClassifier(random_state=42)
        self.callback_calls = []
        
        classifier.fit(
            self.X_cls, self.y_cls,
            population_size=10,
            max_iter=3,
            progress_callback=self.progress_callback
        )
        
        # Check callback was called
        self.assertEqual(len(self.callback_calls), 4)  # max_iter + 1
        
        # Check callback data structure
        for call in self.callback_calls:
            self.assertIn('generation', call)
            self.assertIn('best_fitness', call)
            self.assertIn('avg_fitness', call)
            self.assertIsInstance(call['best_fitness'], (int, float))
            self.assertIsInstance(call['avg_fitness'], (int, float))
    
    def test_regressor_progress_callback(self):
        """Test GATreeRegressor with progress callback"""
        regressor = GATreeRegressor(random_state=42)
        self.callback_calls = []
        
        regressor.fit(
            self.X_reg, self.y_reg,
            population_size=10,
            max_iter=3,
            progress_callback=self.progress_callback
        )
        
        # Check callback was called
        self.assertEqual(len(self.callback_calls), 4)  # max_iter + 1
        
        # Check callback data structure
        for call in self.callback_calls:
            self.assertIn('generation', call)
            self.assertIn('best_fitness', call)
            self.assertIn('avg_fitness', call)
    
    def test_action_selector_progress_callback(self):
        """Test GATreeActionSelector with progress callback"""
        # Simple reward function
        def simple_reward(state, action, y_data, timestep):
            return 1.0 if action == 0 else 0.0
        
        action_selector = GATreeActionSelector(
            action_space=['action_0', 'action_1'],
            reward_function=simple_reward,
            random_state=42
        )
        self.callback_calls = []
        
        # Create simple time series data
        X_ts = pd.DataFrame(np.random.randn(20, 3), columns=['f1', 'f2', 'f3'])
        Y_ts = pd.DataFrame({'dummy': np.ones(20)})
        
        action_selector.fit(
            X_ts, Y_ts,
            population_size=10,
            max_iter=3,
            progress_callback=self.progress_callback
        )
        
        # Check callback was called
        self.assertEqual(len(self.callback_calls), 4)  # max_iter + 1
        
        # Check callback data structure (action selector includes best_reward)
        for call in self.callback_calls:
            self.assertIn('generation', call)
            self.assertIn('best_fitness', call)
            self.assertIn('avg_fitness', call)
            self.assertIn('best_reward', call)
    
    def test_classifier_error_handling(self):
        """Test GATreeClassifier error handling"""
        classifier = GATreeClassifier(random_state=42)
        
        # Test negative sample weights
        with self.assertRaises(RuntimeError):
            negative_weights = np.array([-1.0, 1.0, 1.0])
            classifier.fit(self.X_cls[:3], self.y_cls[:3], sample_weight=negative_weights, max_iter=1)
        
        # Test wrong sample weight length
        with self.assertRaises(RuntimeError):
            wrong_weights = np.array([1.0, 1.0])  # Too short
            classifier.fit(self.X_cls, self.y_cls, sample_weight=wrong_weights, max_iter=1)
        
        # Test zero sum sample weights
        with self.assertRaises(RuntimeError):
            zero_weights = np.zeros(len(self.X_cls))
            classifier.fit(self.X_cls, self.y_cls, sample_weight=zero_weights, max_iter=1)
        
        # Test invalid population size
        with self.assertRaises(RuntimeError):
            classifier.fit(self.X_cls, self.y_cls, population_size=-1, max_iter=1)
        
        # Test invalid mutation probability
        with self.assertRaises(RuntimeError):
            classifier.fit(self.X_cls, self.y_cls, mutation_probability=1.5, max_iter=1)
    
    def test_regressor_error_handling(self):
        """Test GATreeRegressor error handling"""
        regressor = GATreeRegressor(random_state=42)
        
        # Test negative sample weights
        with self.assertRaises(RuntimeError):
            negative_weights = np.array([-1.0, 1.0, 1.0])
            regressor.fit(self.X_reg[:3], self.y_reg[:3], sample_weight=negative_weights, max_iter=1)
        
        # Test invalid elite size
        with self.assertRaises(RuntimeError):
            regressor.fit(self.X_reg, self.y_reg, population_size=5, elite_size=10, max_iter=1)
    
    def test_action_selector_error_handling(self):
        """Test GATreeActionSelector error handling"""
        def simple_reward(state, action, y_data, timestep):
            return 1.0
        
        action_selector = GATreeActionSelector(
            action_space=['action_0', 'action_1'],
            reward_function=simple_reward,
            random_state=42
        )
        
        X_ts = pd.DataFrame(np.random.randn(10, 3), columns=['f1', 'f2', 'f3'])
        Y_ts = pd.DataFrame({'dummy': np.ones(10)})
        
        # Test invalid selection tournament size
        with self.assertRaises(RuntimeError):
            action_selector.fit(X_ts, Y_ts, selection_tournament_size=-1, max_iter=1)
    
    def test_sample_weight_consistency(self):
        """Test that sample weights are consistently applied across all methods"""
        # Test that all methods now accept sample_weight parameter
        classifier = GATreeClassifier(random_state=42)
        regressor = GATreeRegressor(random_state=42)
        
        # Both should accept sample_weight without error
        classifier.fit(self.X_cls, self.y_cls, sample_weight=self.sample_weights, 
                      population_size=5, max_iter=1)
        regressor.fit(self.X_reg, self.y_reg, sample_weight=self.sample_weights, 
                     population_size=5, max_iter=1)
        
        # Check that sample_weight is stored
        self.assertTrue(hasattr(classifier, 'sample_weight'))
        self.assertTrue(hasattr(regressor, 'sample_weight'))
    
    def test_callback_error_handling(self):
        """Test that callback errors don't crash training"""
        def failing_callback(generation, best_fitness, avg_fitness, best_reward=None):
            if generation == 1:
                raise Exception("Callback error")
        
        classifier = GATreeClassifier(random_state=42)
        
        # Should not raise exception despite callback failure
        classifier.fit(
            self.X_cls, self.y_cls,
            population_size=5,
            max_iter=2,
            progress_callback=failing_callback
        )
        
        # Training should complete successfully
        self.assertIsNotNone(classifier._tree)


if __name__ == '__main__':
    unittest.main()