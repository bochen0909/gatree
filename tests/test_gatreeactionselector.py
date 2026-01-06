"""
Tests for GATreeActionSelector class.
"""

import unittest
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from gatree.methods.gatreeactionselector import GATreeActionSelector


class TestGATreeActionSelector(unittest.TestCase):
    """Test cases for GATreeActionSelector."""
    
    def setUp(self):
        """Set up test data."""
        # Generate small synthetic time series data
        np.random.seed(42)
        n_timesteps = 50
        n_features = 3
        
        # Create time series features
        X = np.random.randn(n_timesteps, n_features)
        self.X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
        
        # Create reward calculation data
        self.Y = pd.DataFrame({
            'value': np.random.randn(n_timesteps),
            'cost': np.random.uniform(0.5, 2.0, n_timesteps)
        })
        
        # Define simple action space and reward function
        self.action_space = ['action_0', 'action_1', 'action_2']
        
        def simple_reward_function(state, action, y_data, timestep):
            """Simple reward function for testing."""
            base_reward = y_data['value']
            cost = y_data['cost']
            
            if action == 'action_0':
                return base_reward - cost
            elif action == 'action_1':
                return base_reward * 0.5 - cost * 0.5
            else:  # action_2
                return -cost * 0.1
        
        self.reward_function = simple_reward_function
    
    def test_initialization(self):
        """Test GATreeActionSelector initialization."""
        selector = GATreeActionSelector(
            action_space=self.action_space,
            reward_function=self.reward_function,
            max_depth=5,
            discount_factor=0.9,
            random_state=42
        )
        
        self.assertEqual(selector.action_space, self.action_space)
        self.assertEqual(selector.reward_function, self.reward_function)
        self.assertEqual(selector.max_depth, 5)
        self.assertEqual(selector.discount_factor, 0.9)
        self.assertEqual(selector.action_count, 3)
        self.assertIsNone(selector._tree)
        self.assertEqual(len(selector._best_fitness), 0)
        self.assertEqual(len(selector._avg_fitness), 0)
        self.assertEqual(len(selector._best_rewards), 0)
    
    def test_fit_basic(self):
        """Test basic fitting functionality."""
        selector = GATreeActionSelector(
            action_space=self.action_space,
            reward_function=self.reward_function,
            max_depth=3,
            random_state=42
        )
        
        # Fit with minimal parameters for quick test
        result = selector.fit(
            self.X, 
            self.Y,
            population_size=5,
            max_iter=3,
            mutation_probability=0.1
        )
        
        # Check that fitting returns self
        self.assertEqual(result, selector)
        
        # Check that tree was created
        self.assertIsNotNone(selector._tree)
        
        # Check that fitness tracking works
        self.assertEqual(len(selector._best_fitness), 4)  # max_iter + 1
        self.assertEqual(len(selector._avg_fitness), 4)
        self.assertEqual(len(selector._best_rewards), 4)
        
        # Check that fitness values are reasonable
        self.assertTrue(all(isinstance(f, (int, float)) for f in selector._best_fitness))
        self.assertTrue(all(isinstance(r, (int, float)) for r in selector._best_rewards))
    
    def test_predict_actions(self):
        """Test action prediction functionality."""
        selector = GATreeActionSelector(
            action_space=self.action_space,
            reward_function=self.reward_function,
            max_depth=3,
            random_state=42
        )
        
        # Fit the model
        selector.fit(
            self.X, 
            self.Y,
            population_size=5,
            max_iter=3
        )
        
        # Make predictions
        actions = selector.predict_actions(self.X)
        
        # Check prediction shape and type
        self.assertEqual(len(actions), len(self.X))
        self.assertTrue(all(action in self.action_space for action in actions))
        
        # Test action indices prediction
        action_indices = selector.predict_action_indices(self.X)
        self.assertEqual(len(action_indices), len(self.X))
        self.assertTrue(all(0 <= idx < len(self.action_space) for idx in action_indices))
        
        # Check consistency between actions and indices
        for action, idx in zip(actions, action_indices):
            self.assertEqual(action, self.action_space[idx])
    
    def test_simulate_rewards(self):
        """Test reward simulation functionality."""
        selector = GATreeActionSelector(
            action_space=self.action_space,
            reward_function=self.reward_function,
            max_depth=3,
            random_state=42
        )
        
        # Fit the model
        selector.fit(
            self.X, 
            self.Y,
            population_size=5,
            max_iter=3
        )
        
        # Simulate rewards
        total_reward, individual_rewards, actions = selector.simulate_rewards(self.X, self.Y)
        
        # Check return types and shapes
        self.assertIsInstance(total_reward, (int, float))
        self.assertEqual(len(individual_rewards), len(self.X))
        self.assertEqual(len(actions), len(self.X))
        
        # Check that all actions are valid
        self.assertTrue(all(action in self.action_space for action in actions))
        
        # Check that individual rewards sum correctly (considering discount factor)
        expected_total = sum(reward * (selector.discount_factor ** i) 
                           for i, reward in enumerate(individual_rewards))
        self.assertAlmostEqual(total_reward, expected_total, places=5)
    
    def test_fitness_function(self):
        """Test the default fitness function."""
        selector = GATreeActionSelector(
            action_space=self.action_space,
            reward_function=self.reward_function,
            random_state=42
        )
        
        # Create a mock tree node
        from gatree.tree.node import Node
        mock_tree = Node()
        mock_tree.y_pred = [0, 1, 2, 0, 1]  # Action indices
        mock_tree.size = lambda: 5
        
        # Mock predict_one method
        def mock_predict_one(state, train=False):
            return np.random.randint(0, len(self.action_space))
        mock_tree.predict_one = mock_predict_one
        
        # Test fitness calculation
        fitness = selector.default_fitness_function(
            mock_tree, 
            self.X.iloc[:5], 
            self.Y.iloc[:5],
            self.action_space,
            self.reward_function,
            discount_factor=1.0
        )
        
        # Fitness should be a number
        self.assertIsInstance(fitness, (int, float))
        
        # Test with empty predictions (should return inf)
        empty_tree = Node()
        empty_tree.y_pred = []
        empty_tree.size = lambda: 1
        
        fitness_empty = selector.default_fitness_function(
            empty_tree, self.X, self.Y, self.action_space, self.reward_function
        )
        self.assertEqual(fitness_empty, float('inf'))
    
    def test_action_distribution(self):
        """Test action distribution calculation."""
        selector = GATreeActionSelector(
            action_space=self.action_space,
            reward_function=self.reward_function,
            max_depth=3,
            random_state=42
        )
        
        # Fit the model
        selector.fit(
            self.X, 
            self.Y,
            population_size=5,
            max_iter=3
        )
        
        # Get action distribution
        distribution = selector.get_action_distribution(self.X)
        
        # Check that all actions are represented
        self.assertEqual(set(distribution.keys()), set(self.action_space))
        
        # Check that counts sum to total timesteps
        total_count = sum(distribution.values())
        self.assertEqual(total_count, len(self.X))
        
        # Check that all counts are non-negative integers
        self.assertTrue(all(isinstance(count, int) and count >= 0 
                          for count in distribution.values()))
    
    def test_discount_factor(self):
        """Test discount factor functionality."""
        # Test with different discount factors
        for discount in [0.5, 0.9, 1.0]:
            selector = GATreeActionSelector(
                action_space=self.action_space,
                reward_function=self.reward_function,
                discount_factor=discount,
                random_state=42
            )
            
            selector.fit(
                self.X, 
                self.Y,
                population_size=5,
                max_iter=2
            )
            
            total_reward, individual_rewards, _ = selector.simulate_rewards(self.X, self.Y)
            
            # Calculate expected total with discount
            expected_total = sum(reward * (discount ** i) 
                               for i, reward in enumerate(individual_rewards))
            
            self.assertAlmostEqual(total_reward, expected_total, places=5)
    
    def test_custom_reward_function(self):
        """Test with custom reward function."""
        def custom_reward(state, action, y_data, timestep):
            """Custom reward function that depends on timestep."""
            return timestep * 0.1 if action == 'action_0' else -timestep * 0.1
        
        selector = GATreeActionSelector(
            action_space=self.action_space,
            reward_function=custom_reward,
            random_state=42
        )
        
        selector.fit(
            self.X, 
            self.Y,
            population_size=5,
            max_iter=2
        )
        
        # Test simulation with custom function
        total_reward, individual_rewards, actions = selector.simulate_rewards(
            self.X, self.Y, custom_reward
        )
        
        # Check that rewards follow the custom function pattern
        self.assertIsInstance(total_reward, (int, float))
        self.assertEqual(len(individual_rewards), len(self.X))
    
    def test_error_handling(self):
        """Test error handling in various scenarios."""
        selector = GATreeActionSelector(
            action_space=self.action_space,
            reward_function=self.reward_function,
            random_state=42
        )
        
        # Test prediction before fitting
        with self.assertRaises(ValueError):
            selector.predict_actions(self.X)
        
        with self.assertRaises(ValueError):
            selector.predict_action_indices(self.X)
        
        # Test with problematic reward function
        def error_reward(state, action, y_data, timestep):
            if timestep == 2:
                raise ValueError("Test error")
            return 1.0
        
        error_selector = GATreeActionSelector(
            action_space=self.action_space,
            reward_function=error_reward,
            random_state=42
        )
        
        # Should handle errors gracefully during fitting
        error_selector.fit(
            self.X.iloc[:5], 
            self.Y.iloc[:5],
            population_size=3,
            max_iter=1
        )
        
        # Tree should still be created despite errors
        self.assertIsNotNone(error_selector._tree)
    
    def test_string_representations(self):
        """Test string representations of the selector."""
        selector = GATreeActionSelector(
            action_space=self.action_space,
            reward_function=self.reward_function,
            random_state=42
        )
        
        # Test unfitted selector
        str_repr = str(selector)
        self.assertIn("unfitted", str_repr)
        self.assertIn(str(self.action_space), str_repr)
        
        # Test fitted selector
        selector.fit(
            self.X, 
            self.Y,
            population_size=3,
            max_iter=1
        )
        
        str_repr = str(selector)
        self.assertIn("depth=", str_repr)
        self.assertIn("size=", str_repr)
        self.assertIn(str(self.action_space), str_repr)
        
        # Test repr
        repr_str = repr(selector)
        self.assertEqual(str_repr, repr_str)
    
    def test_small_action_space(self):
        """Test with minimal action space."""
        small_action_space = ['single_action']
        
        def simple_reward(state, action, y_data, timestep):
            return 1.0
        
        selector = GATreeActionSelector(
            action_space=small_action_space,
            reward_function=simple_reward,
            random_state=42
        )
        
        selector.fit(
            self.X.iloc[:10], 
            self.Y.iloc[:10],
            population_size=3,
            max_iter=2
        )
        
        actions = selector.predict_actions(self.X.iloc[:10])
        
        # All actions should be the single action
        self.assertTrue(all(action == 'single_action' for action in actions))
    
    def test_large_action_space(self):
        """Test with larger action space."""
        large_action_space = [f'action_{i}' for i in range(10)]
        
        def varied_reward(state, action, y_data, timestep):
            action_idx = large_action_space.index(action)
            return action_idx * 0.1 - 0.5  # Reward increases with action index
        
        selector = GATreeActionSelector(
            action_space=large_action_space,
            reward_function=varied_reward,
            random_state=42
        )
        
        selector.fit(
            self.X, 
            self.Y,
            population_size=10,
            max_iter=5
        )
        
        actions = selector.predict_actions(self.X)
        
        # Check that actions are from the correct space
        self.assertTrue(all(action in large_action_space for action in actions))
        
        # Check action distribution
        distribution = selector.get_action_distribution(self.X)
        self.assertEqual(len(distribution), len(large_action_space))


if __name__ == '__main__':
    unittest.main()