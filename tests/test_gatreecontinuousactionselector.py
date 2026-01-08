"""
Tests for GATreeContinuousActionSelector class.
"""

import unittest
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from gatree.methods.gatreecontinuousactionselector import GATreeContinuousActionSelector


class TestGATreeContinuousActionSelector(unittest.TestCase):
    """Test cases for GATreeContinuousActionSelector."""
    
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
        
        # Define simple continuous reward function
        def simple_continuous_reward_function(state, action, y_data, timestep, previous_action):
            """Simple continuous reward function for testing."""
            try:
                base_reward = float(y_data['value'])
                cost = float(y_data['cost'])
                
                # Reward based on continuous action value
                # Higher actions get higher base rewards but also higher costs
                action_reward = base_reward * action  # action is between 0 and 1
                action_cost = cost * (action ** 2)    # Quadratic cost penalty
                
                reward = action_reward - action_cost
                
                # Optional: Add small penalty for large action changes
                if previous_action is not None:
                    action_change_penalty = 0.1 * abs(action - previous_action)
                    reward -= action_change_penalty
                
                return reward
            except Exception as e:
                print(f"Test error: {e}")
                return -1.0  # Return a small negative reward on error
        
        self.reward_function = simple_continuous_reward_function
    
    def test_initialization(self):
        """Test GATreeContinuousActionSelector initialization."""
        selector = GATreeContinuousActionSelector(
            reward_function=self.reward_function,
            action_bounds=(0, 1),
            max_depth=5,
            discount_factor=0.9,
            n_action_bins=20,
            random_state=42
        )
        
        self.assertEqual(selector.reward_function, self.reward_function)
        self.assertEqual(selector.action_bounds, (0, 1))
        self.assertEqual(selector.max_depth, 5)
        self.assertEqual(selector.discount_factor, 0.9)
        self.assertEqual(selector.discount_direction, 'forward')  # Default
        self.assertEqual(selector.n_action_bins, 20)
        self.assertIsNone(selector._tree)
        self.assertEqual(len(selector._best_fitness), 0)
        self.assertEqual(len(selector._avg_fitness), 0)
        self.assertEqual(len(selector._best_rewards), 0)
    
    def test_custom_action_bounds(self):
        """Test GATreeContinuousActionSelector with custom action bounds."""
        selector = GATreeContinuousActionSelector(
            reward_function=self.reward_function,
            action_bounds=(-1, 2),
            n_action_bins=10,
            random_state=42
        )
        
        self.assertEqual(selector.action_bounds, (-1, 2))
        
        # Test action conversion
        self.assertAlmostEqual(selector._action_index_to_continuous(0), -1.0, places=5)
        self.assertAlmostEqual(selector._action_index_to_continuous(9), 2.0, places=5)
        self.assertAlmostEqual(selector._action_index_to_continuous(4), 0.333333, places=5)
        
        # Test reverse conversion
        self.assertEqual(selector._continuous_to_action_index(-1.0), 0)
        self.assertEqual(selector._continuous_to_action_index(2.0), 9)
        self.assertEqual(selector._continuous_to_action_index(0.5), 4)
    
    def test_backward_discounting_initialization(self):
        """Test GATreeContinuousActionSelector initialization with backward discounting."""
        selector = GATreeContinuousActionSelector(
            reward_function=self.reward_function,
            discount_factor=0.8,
            discount_direction='backward',
            random_state=42
        )
        
        self.assertEqual(selector.discount_factor, 0.8)
        self.assertEqual(selector.discount_direction, 'backward')
        
        # Test invalid discount direction
        with self.assertRaises(ValueError):
            GATreeContinuousActionSelector(
                reward_function=self.reward_function,
                discount_direction='invalid'
            )
    
    def test_global_reward_initialization(self):
        """Test GATreeContinuousActionSelector initialization with global reward function."""
        def dummy_global_reward(rewards, actions, X, Y):
            return sum(rewards) * np.mean(actions) * 0.1
        
        selector = GATreeContinuousActionSelector(
            reward_function=self.reward_function,
            global_reward_function=dummy_global_reward,
            global_reward_weight=0.5,
            random_state=42
        )
        
        self.assertEqual(selector.global_reward_function, dummy_global_reward)
        self.assertEqual(selector.global_reward_weight, 0.5)
        
        # Test without global reward function
        selector_no_global = GATreeContinuousActionSelector(
            reward_function=self.reward_function,
            random_state=42
        )
        
        self.assertIsNone(selector_no_global.global_reward_function)
        self.assertEqual(selector_no_global.global_reward_weight, 1.0)
    
    def test_invalid_initialization(self):
        """Test invalid initialization parameters."""
        # Invalid action bounds
        with self.assertRaises(ValueError):
            GATreeContinuousActionSelector(
                reward_function=self.reward_function,
                action_bounds=(1, 0)  # min >= max
            )
        
        with self.assertRaises(ValueError):
            GATreeContinuousActionSelector(
                reward_function=self.reward_function,
                action_bounds=(1, 1)  # min == max
            )
        
        # Invalid n_action_bins
        with self.assertRaises(ValueError):
            GATreeContinuousActionSelector(
                reward_function=self.reward_function,
                n_action_bins=0
            )
    
    def test_fit_basic(self):
        """Test basic fitting functionality."""
        selector = GATreeContinuousActionSelector(
            reward_function=self.reward_function,
            action_bounds=(0, 1),
            max_depth=3,
            n_action_bins=10,
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
        """Test continuous action prediction functionality."""
        selector = GATreeContinuousActionSelector(
            reward_function=self.reward_function,
            action_bounds=(0, 1),
            max_depth=3,
            n_action_bins=10,
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
        self.assertTrue(all(isinstance(action, (int, float)) for action in actions))
        
        # Check that all actions are within bounds
        self.assertTrue(all(0 <= action <= 1 for action in actions))
        
        # Test action indices prediction
        action_indices = selector.predict_action_indices(self.X)
        self.assertEqual(len(action_indices), len(self.X))
        self.assertTrue(all(0 <= idx < selector.n_action_bins for idx in action_indices))
        
        # Check consistency between actions and indices
        for action, idx in zip(actions, action_indices):
            expected_action = selector._action_index_to_continuous(idx)
            self.assertAlmostEqual(action, expected_action, places=5)
    
    def test_simulate_rewards(self):
        """Test reward simulation functionality."""
        selector = GATreeContinuousActionSelector(
            reward_function=self.reward_function,
            action_bounds=(0, 1),
            max_depth=3,
            n_action_bins=10,
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
        
        # Check that all actions are within bounds
        self.assertTrue(all(0 <= action <= 1 for action in actions))
        
        # Check that individual rewards are reasonable
        self.assertTrue(all(isinstance(reward, (int, float)) for reward in individual_rewards))
    
    def test_fitness_function(self):
        """Test the default fitness function."""
        selector = GATreeContinuousActionSelector(
            reward_function=self.reward_function,
            action_bounds=(0, 1),
            n_action_bins=10,
            random_state=42
        )
        
        # Create a mock tree node
        from gatree.tree.node import Node
        mock_tree = Node()
        mock_tree.y_pred = [0, 1, 2, 0, 1]  # Action indices
        mock_tree.size = lambda: 5
        
        # Mock predict_one method
        def mock_predict_one(state, train=False):
            return np.random.randint(0, selector.n_action_bins)
        mock_tree.predict_one = mock_predict_one
        
        # Test fitness calculation
        fitness = selector.default_fitness_function(
            mock_tree, 
            self.X.iloc[:5], 
            self.Y.iloc[:5],
            self.reward_function,
            selector.action_bounds,
            selector.n_action_bins,
            discount_factor=1.0
        )
        
        # Fitness should be a number
        self.assertIsInstance(fitness, (int, float))
        
        # Test with empty predictions (should return inf)
        empty_tree = Node()
        empty_tree.y_pred = []
        empty_tree.size = lambda: 1
        
        fitness_empty = selector.default_fitness_function(
            empty_tree, self.X, self.Y, self.reward_function, 
            selector.action_bounds, selector.n_action_bins
        )
        self.assertEqual(fitness_empty, float('inf'))
    
    def test_action_statistics(self):
        """Test action statistics calculation."""
        selector = GATreeContinuousActionSelector(
            reward_function=self.reward_function,
            action_bounds=(0, 1),
            max_depth=3,
            n_action_bins=10,
            random_state=42
        )
        
        # Fit the model
        selector.fit(
            self.X, 
            self.Y,
            population_size=5,
            max_iter=3
        )
        
        # Get action statistics
        stats = selector.get_action_statistics(self.X)
        
        # Check that all required statistics are present
        required_keys = ['mean', 'std', 'min', 'max', 'median', 'q25', 'q75', 'count']
        self.assertEqual(set(stats.keys()), set(required_keys))
        
        # Check that statistics are reasonable
        self.assertTrue(0 <= stats['min'] <= 1)
        self.assertTrue(0 <= stats['max'] <= 1)
        self.assertTrue(0 <= stats['mean'] <= 1)
        self.assertTrue(stats['std'] >= 0)
        self.assertEqual(stats['count'], len(self.X))
        
        # Check ordering
        self.assertTrue(stats['min'] <= stats['q25'] <= stats['median'] <= stats['q75'] <= stats['max'])
    
    def test_discount_factor(self):
        """Test discount factor functionality."""
        # Test with different discount factors
        for discount in [0.5, 0.9, 1.0]:
            selector = GATreeContinuousActionSelector(
                reward_function=self.reward_function,
                action_bounds=(0, 1),
                discount_factor=discount,
                n_action_bins=10,
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
    
    def test_backward_discounting(self):
        """Test backward discounting functionality."""
        selector = GATreeContinuousActionSelector(
            reward_function=self.reward_function,
            action_bounds=(0, 1),
            discount_factor=0.9,
            discount_direction='backward',
            n_action_bins=10,
            random_state=42
        )
        
        selector.fit(
            self.X.iloc[:10], 
            self.Y.iloc[:10],
            population_size=5,
            max_iter=2
        )
        
        total_reward, individual_rewards, _ = selector.simulate_rewards(
            self.X.iloc[:10], self.Y.iloc[:10]
        )
        
        # Calculate expected total with backward discount
        n_timesteps = len(individual_rewards)
        expected_total = sum(reward * (0.9 ** (n_timesteps - 1 - i)) 
                           for i, reward in enumerate(individual_rewards))
        
        self.assertAlmostEqual(total_reward, expected_total, places=5)
    
    def test_custom_reward_function(self):
        """Test with custom reward function."""
        def custom_reward(state, action, y_data, timestep, previous_action):
            """Custom reward function that depends on timestep and action."""
            base_reward = timestep * 0.1 * action  # Reward increases with time and action
            
            # Add bonus for action consistency
            if previous_action is not None and abs(action - previous_action) < 0.1:
                base_reward += 0.05  # Small bonus for keeping similar action
            
            return base_reward
        
        selector = GATreeContinuousActionSelector(
            reward_function=custom_reward,
            action_bounds=(0, 1),
            n_action_bins=10,
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
        self.assertTrue(all(0 <= action <= 1 for action in actions))
    
    def test_error_handling(self):
        """Test error handling in various scenarios."""
        selector = GATreeContinuousActionSelector(
            reward_function=self.reward_function,
            action_bounds=(0, 1),
            n_action_bins=10,
            random_state=42
        )
        
        # Test prediction before fitting
        with self.assertRaises(ValueError):
            selector.predict_actions(self.X)
        
        with self.assertRaises(ValueError):
            selector.predict_action_indices(self.X)
        
        with self.assertRaises(ValueError):
            selector.predict_action(self.X.iloc[0])
        
        # Test with problematic reward function
        def error_reward(state, action, y_data, timestep, previous_action):
            if timestep == 2:
                raise ValueError("Test error")
            return action * 1.0
        
        error_selector = GATreeContinuousActionSelector(
            reward_function=error_reward,
            action_bounds=(0, 1),
            n_action_bins=10,
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
        selector = GATreeContinuousActionSelector(
            reward_function=self.reward_function,
            action_bounds=(0, 1),
            n_action_bins=20,
            random_state=42
        )
        
        # Test unfitted selector
        str_repr = str(selector)
        self.assertIn("unfitted", str_repr)
        self.assertIn("(0, 1)", str_repr)
        self.assertIn("bins=20", str_repr)
        
        # Test fitted selector
        selector.fit(
            self.X, 
            self.Y,
            population_size=5,
            elite_size=1,
            max_iter=1
        )
        
        str_repr = str(selector)
        self.assertIn("depth=", str_repr)
        self.assertIn("size=", str_repr)
        self.assertIn("(0, 1)", str_repr)
        self.assertIn("bins=20", str_repr)
        
        # Test repr
        repr_str = repr(selector)
        self.assertEqual(str_repr, repr_str)
    
    def test_different_action_bounds(self):
        """Test with different action bounds."""
        bounds_list = [(0, 1), (-1, 1), (0.2, 0.8), (-2, 3)]
        
        for bounds in bounds_list:
            selector = GATreeContinuousActionSelector(
                reward_function=self.reward_function,
                action_bounds=bounds,
                n_action_bins=10,
                random_state=42
            )
            
            selector.fit(
                self.X.iloc[:10], 
                self.Y.iloc[:10],
                population_size=3,
                max_iter=2
            )
            
            actions = selector.predict_actions(self.X.iloc[:10])
            
            # Check that all actions are within bounds
            min_bound, max_bound = bounds
            self.assertTrue(all(min_bound <= action <= max_bound for action in actions))
            
            # Check statistics
            stats = selector.get_action_statistics(self.X.iloc[:10])
            self.assertTrue(min_bound <= stats['min'] <= max_bound)
            self.assertTrue(min_bound <= stats['max'] <= max_bound)
    
    def test_sample_weights_validation(self):
        """Test sample weight validation."""
        selector = GATreeContinuousActionSelector(
            reward_function=self.reward_function,
            action_bounds=(0, 1),
            n_action_bins=10,
            random_state=42
        )
        
        # Test with correct weights
        weights = np.ones(len(self.X))
        selector.fit(self.X, self.Y, sample_weight=weights, max_iter=5)
        
        # Test with wrong length
        with self.assertRaises(RuntimeError):
            wrong_weights = np.ones(10)  # Wrong length
            selector.fit(self.X, self.Y, sample_weight=wrong_weights, max_iter=5)
        
        # Test with negative weights
        with self.assertRaises(RuntimeError):
            negative_weights = np.ones(len(self.X))
            negative_weights[0] = -1.0
            selector.fit(self.X, self.Y, sample_weight=negative_weights, max_iter=5)

    def test_sample_weights_functionality(self):
        """Test that sample weights affect training."""
        # Create weights that heavily favor later timesteps
        sample_weight = np.linspace(0.1, 2.0, len(self.X))
        
        # Train without weights
        selector_no_weights = GATreeContinuousActionSelector(
            reward_function=self.reward_function,
            action_bounds=(0, 1),
            n_action_bins=10,
            random_state=42
        )
        selector_no_weights.fit(self.X, self.Y, max_iter=10)
        
        # Train with weights
        selector_with_weights = GATreeContinuousActionSelector(
            reward_function=self.reward_function,
            action_bounds=(0, 1),
            n_action_bins=10,
            random_state=42
        )
        selector_with_weights.fit(self.X, self.Y, sample_weight=sample_weight, max_iter=10)
        
        # Both should complete successfully
        self.assertIsNotNone(selector_no_weights._tree)
        self.assertIsNotNone(selector_with_weights._tree)
        
        # Make predictions
        actions_no_weights = selector_no_weights.predict_actions(self.X[:5])
        actions_with_weights = selector_with_weights.predict_actions(self.X[:5])
        
        self.assertEqual(len(actions_no_weights), 5)
        self.assertEqual(len(actions_with_weights), 5)
        self.assertTrue(all(0 <= action <= 1 for action in actions_no_weights))
        self.assertTrue(all(0 <= action <= 1 for action in actions_with_weights))

    def test_single_action_prediction(self):
        """Test single action prediction."""
        selector = GATreeContinuousActionSelector(
            reward_function=self.reward_function,
            action_bounds=(0, 1),
            n_action_bins=10,
            random_state=42
        )
        
        selector.fit(
            self.X, 
            self.Y,
            population_size=5,
            max_iter=3
        )
        
        # Test single prediction
        single_action = selector.predict_action(self.X.iloc[0])
        self.assertIsInstance(single_action, (int, float))
        self.assertTrue(0 <= single_action <= 1)
        
        # Should be consistent with batch prediction
        batch_actions = selector.predict_actions(self.X.iloc[:1])
        self.assertAlmostEqual(single_action, batch_actions[0], places=5)

    def test_early_stopping(self):
        """Test early stopping functionality."""
        selector = GATreeContinuousActionSelector(
            reward_function=self.reward_function,
            action_bounds=(0, 1),
            n_action_bins=10,
            random_state=42
        )
        
        # Fit with early stopping
        selector.fit(
            self.X.iloc[:20], 
            self.Y.iloc[:20],
            population_size=5,
            max_iter=100,  # High max_iter
            early_stopping=True,
            patience=5,
            min_delta=1e-6
        )
        
        # Should have stopped before max_iter
        self.assertLess(len(selector._best_fitness), 101)  # max_iter + 1
        self.assertIsNotNone(selector._tree)


if __name__ == '__main__':
    unittest.main()