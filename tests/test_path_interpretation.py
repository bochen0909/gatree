"""
Tests for the path interpretation functionality in GATree.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression

from gatree.methods import GATreeClassifier, GATreeRegressor, GATreeActionSelector, GATreeContinuousActionSelector
from gatree.utils import format_decision_path, print_decision_path, analyze_decision_paths


class TestPathInterpretation:
    """Test cases for path interpretation functionality."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Classification data
        X_class, y_class = make_classification(n_samples=50, n_features=3, n_informative=2, 
                                             n_redundant=0, n_classes=2, random_state=42)
        self.X_class = pd.DataFrame(X_class, columns=['f1', 'f2', 'f3'])
        self.y_class = pd.Series(y_class)
        
        # Regression data
        X_reg, y_reg = make_regression(n_samples=50, n_features=3, noise=0.1, random_state=42)
        self.X_reg = pd.DataFrame(X_reg, columns=['f1', 'f2', 'f3'])
        self.y_reg = pd.Series(y_reg)
        
        # Time series data for action selection
        self.X_ts = pd.DataFrame(np.random.randn(20, 3), columns=['f1', 'f2', 'f3'])
        self.Y_ts = pd.DataFrame({'price': np.random.randn(20) * 10 + 100})
    
    def test_classifier_predict_with_path(self):
        """Test path interpretation for classifier."""
        classifier = GATreeClassifier(max_depth=3, random_state=42)
        classifier.fit(self.X_class, self.y_class, population_size=10, max_iter=5)
        
        # Test batch prediction with paths
        predictions, paths = classifier.predict_with_path(self.X_class.head(3))
        
        assert len(predictions) == 3
        assert len(paths) == 3
        assert all(isinstance(p, (int, np.integer)) for p in predictions)
        assert all(isinstance(path, list) for path in paths)
        
        # Check path structure
        for path in paths:
            assert len(path) > 0
            for step in path:
                assert isinstance(step, dict)
                assert 'node_type' in step
                assert step['node_type'] in ['internal', 'leaf', 'missing_child', 'error']
    
    def test_classifier_predict_single_with_path(self):
        """Test single prediction with path for classifier."""
        classifier = GATreeClassifier(max_depth=3, random_state=42)
        classifier.fit(self.X_class, self.y_class, population_size=10, max_iter=5)
        
        # Test single prediction with path
        prediction, path = classifier.predict_single_with_path(self.X_class.iloc[0])
        
        assert isinstance(prediction, (int, np.integer))
        assert isinstance(path, list)
        assert len(path) > 0
        
        # Check that path ends with a leaf
        assert path[-1]['node_type'] == 'leaf'
        assert path[-1]['predicted_value'] == prediction
    
    def test_regressor_predict_with_path(self):
        """Test path interpretation for regressor."""
        regressor = GATreeRegressor(max_depth=3, random_state=42)
        regressor.fit(self.X_reg, self.y_reg, population_size=10, max_iter=5)
        
        # Test batch prediction with paths
        predictions, paths = regressor.predict_with_path(self.X_reg.head(2))
        
        assert len(predictions) == 2
        assert len(paths) == 2
        assert all(isinstance(p, (int, float, np.number)) for p in predictions)
        assert all(isinstance(path, list) for path in paths)
    
    def test_action_selector_predict_with_path(self):
        """Test path interpretation for action selector."""
        def simple_reward(state, action, y_data, timestep, previous_action):
            return np.random.randn()
        
        action_space = ['buy', 'sell', 'hold']
        selector = GATreeActionSelector(
            action_space=action_space,
            reward_function=simple_reward,
            max_depth=3,
            random_state=42
        )
        
        selector.fit(self.X_ts, self.Y_ts, population_size=10, max_iter=5)
        
        # Test batch prediction with paths
        actions, paths = selector.predict_actions_with_path(self.X_ts.head(3))
        
        assert len(actions) == 3
        assert len(paths) == 3
        assert all(action in action_space for action in actions)
        assert all(isinstance(path, list) for path in paths)
    
    def test_continuous_action_selector_predict_with_path(self):
        """Test path interpretation for continuous action selector."""
        def simple_reward(state, action, y_data, timestep, previous_action):
            return -abs(action - 0.5)  # Reward closer to 0.5
        
        selector = GATreeContinuousActionSelector(
            reward_function=simple_reward,
            action_bounds=(0, 1),
            max_depth=3,
            n_action_bins=10,
            random_state=42
        )
        
        selector.fit(self.X_ts, self.Y_ts, population_size=10, max_iter=5)
        
        # Test batch prediction with paths
        actions, paths = selector.predict_actions_with_path(self.X_ts.head(2))
        
        assert len(actions) == 2
        assert len(paths) == 2
        assert all(0 <= action <= 1 for action in actions)
        assert all(isinstance(path, list) for path in paths)
    
    def test_format_decision_path(self):
        """Test decision path formatting."""
        # Create a sample path
        path = [
            {
                'node_type': 'internal',
                'feature_index': 0,
                'feature_name': 'feature_0',
                'feature_value': 1.5,
                'threshold': 1.0,
                'condition': 'feature_0 > 1.0',
                'decision': True
            },
            {
                'node_type': 'leaf',
                'predicted_value': 1,
                'leaf_value': 1
            }
        ]
        
        formatted = format_decision_path(path)
        assert isinstance(formatted, str)
        assert 'Step 1:' in formatted
        assert 'Final:' in formatted
        assert '1.5000 > 1.0000? YES' in formatted
        assert 'Predicted value = 1' in formatted
    
    def test_analyze_decision_paths(self):
        """Test decision path analysis."""
        # Create sample paths
        paths = [
            [
                {
                    'node_type': 'internal',
                    'feature_index': 0,
                    'feature_name': 'f1',
                    'feature_value': 1.5,
                    'threshold': 1.0,
                    'decision': True
                },
                {
                    'node_type': 'leaf',
                    'predicted_value': 1,
                    'leaf_value': 1
                }
            ],
            [
                {
                    'node_type': 'internal',
                    'feature_index': 0,
                    'feature_name': 'f1',
                    'feature_value': 0.5,
                    'threshold': 1.0,
                    'decision': False
                },
                {
                    'node_type': 'leaf',
                    'predicted_value': 0,
                    'leaf_value': 0
                }
            ]
        ]
        
        analysis = analyze_decision_paths(paths)
        
        assert analysis['total_paths'] == 2
        assert 'feature_usage' in analysis
        assert 'f1' in analysis['feature_usage']
        assert analysis['feature_usage']['f1'] == 2
        assert analysis['avg_path_length'] == 1.0
        assert len(analysis['leaf_values']) == 2
    
    def test_unfitted_model_error(self):
        """Test that unfitted models raise appropriate errors."""
        classifier = GATreeClassifier()
        
        with pytest.raises(ValueError, match="must be fitted"):
            classifier.predict_with_path(self.X_class.head(1))
        
        with pytest.raises(ValueError, match="must be fitted"):
            classifier.predict_single_with_path(self.X_class.iloc[0])


if __name__ == "__main__":
    pytest.main([__file__])