"""
Example demonstrating the new path interpretation functionality in GATree.

This example shows how to use the predict_with_path methods to understand
how the decision trees make their predictions.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

# GATree imports
from gatree.methods import GATreeClassifier, GATreeRegressor, GATreeActionSelector, GATreeContinuousActionSelector
from gatree.utils import format_decision_path, print_decision_path, analyze_decision_paths, print_path_analysis


def classification_example():
    """Demonstrate path interpretation for classification."""
    print("=" * 60)
    print("CLASSIFICATION EXAMPLE")
    print("=" * 60)
    
    # Generate sample data
    X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y_series = pd.Series(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_series, test_size=0.3, random_state=42)
    
    # Train classifier
    classifier = GATreeClassifier(max_depth=3, random_state=42)
    classifier.fit(X_train, y_train, population_size=20, max_iter=10)
    
    # Make predictions with paths
    predictions, paths = classifier.predict_with_path(X_test.head(3))
    
    print(f"Predictions: {predictions}")
    print(f"Actual values: {y_test.head(3).tolist()}")
    
    # Show individual decision paths
    feature_names = X_df.columns.tolist()
    for i, (pred, path) in enumerate(zip(predictions, paths)):
        print_decision_path(path, feature_names, f"Sample {i+1} Decision Path (Predicted: {pred})")
    
    # Analyze all paths
    all_predictions, all_paths = classifier.predict_with_path(X_test)
    analysis = analyze_decision_paths(all_paths, feature_names)
    print_path_analysis(analysis)


def regression_example():
    """Demonstrate path interpretation for regression."""
    print("=" * 60)
    print("REGRESSION EXAMPLE")
    print("=" * 60)
    
    # Generate sample data
    X, y = make_regression(n_samples=100, n_features=3, noise=0.1, random_state=42)
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y_series = pd.Series(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_series, test_size=0.3, random_state=42)
    
    # Train regressor
    regressor = GATreeRegressor(max_depth=3, random_state=42)
    regressor.fit(X_train, y_train, population_size=20, max_iter=10)
    
    # Make predictions with paths
    predictions, paths = regressor.predict_with_path(X_test.head(2))
    
    print(f"Predictions: {[f'{p:.2f}' for p in predictions]}")
    print(f"Actual values: {[f'{v:.2f}' for v in y_test.head(2).tolist()]}")
    
    # Show individual decision paths
    feature_names = X_df.columns.tolist()
    for i, (pred, path) in enumerate(zip(predictions, paths)):
        print_decision_path(path, feature_names, f"Sample {i+1} Decision Path (Predicted: {pred:.2f})")


def action_selection_example():
    """Demonstrate path interpretation for action selection."""
    print("=" * 60)
    print("ACTION SELECTION EXAMPLE")
    print("=" * 60)
    
    # Generate sample time series data
    np.random.seed(42)
    n_timesteps = 20
    n_features = 3
    
    X = pd.DataFrame(np.random.randn(n_timesteps, n_features), 
                     columns=[f'feature_{i}' for i in range(n_features)])
    Y = pd.DataFrame({'price': np.random.randn(n_timesteps) * 10 + 100})
    
    # Define a simple reward function
    def simple_reward(state, action, y_data, timestep, previous_action):
        if action == 'buy':
            return y_data['price'] * 0.01  # Reward based on price
        elif action == 'sell':
            return -y_data['price'] * 0.005  # Small penalty
        else:  # hold
            return 0.1  # Small positive reward for holding
    
    # Train action selector
    action_space = ['buy', 'sell', 'hold']
    selector = GATreeActionSelector(
        action_space=action_space,
        reward_function=simple_reward,
        max_depth=3,
        random_state=42
    )
    
    selector.fit(X, Y, population_size=20, max_iter=10)
    
    # Make predictions with paths for first few timesteps
    test_X = X.head(3)
    actions, paths = selector.predict_actions_with_path(test_X)
    
    print(f"Predicted actions: {actions}")
    
    # Show individual decision paths
    feature_names = X.columns.tolist()
    for i, (action, path) in enumerate(zip(actions, paths)):
        print_decision_path(path, feature_names, f"Timestep {i+1} Decision Path (Action: {action})")


def continuous_action_example():
    """Demonstrate path interpretation for continuous action selection."""
    print("=" * 60)
    print("CONTINUOUS ACTION SELECTION EXAMPLE")
    print("=" * 60)
    
    # Generate sample time series data
    np.random.seed(42)
    n_timesteps = 15
    n_features = 2
    
    X = pd.DataFrame(np.random.randn(n_timesteps, n_features), 
                     columns=[f'feature_{i}' for i in range(n_features)])
    Y = pd.DataFrame({'target': np.random.randn(n_timesteps)})
    
    # Define a simple reward function for continuous actions
    def continuous_reward(state, action, y_data, timestep, previous_action):
        # Reward is higher when action is closer to the target
        target = (y_data['target'] + 1) / 2  # Normalize to [0, 1]
        return -abs(action - target)  # Negative distance as reward
    
    # Train continuous action selector
    selector = GATreeContinuousActionSelector(
        reward_function=continuous_reward,
        action_bounds=(0, 1),
        max_depth=3,
        n_action_bins=10,
        random_state=42
    )
    
    selector.fit(X, Y, population_size=20, max_iter=10)
    
    # Make predictions with paths for first few timesteps
    test_X = X.head(2)
    actions, paths = selector.predict_actions_with_path(test_X)
    
    print(f"Predicted actions: {[f'{a:.3f}' for a in actions]}")
    
    # Show individual decision paths
    feature_names = X.columns.tolist()
    for i, (action, path) in enumerate(zip(actions, paths)):
        print_decision_path(path, feature_names, f"Timestep {i+1} Decision Path (Action: {action:.3f})")


def single_prediction_example():
    """Demonstrate single prediction with path interpretation."""
    print("=" * 60)
    print("SINGLE PREDICTION EXAMPLE")
    print("=" * 60)
    
    # Generate sample data
    X, y = make_classification(n_samples=50, n_features=3, n_informative=2, n_redundant=0, n_classes=2, random_state=42)
    X_df = pd.DataFrame(X, columns=['temperature', 'humidity', 'pressure'])
    y_series = pd.Series(y)
    
    # Train classifier
    classifier = GATreeClassifier(max_depth=4, random_state=42)
    classifier.fit(X_df, y_series, population_size=15, max_iter=8)
    
    # Make a single prediction with path
    single_instance = X_df.iloc[0]
    prediction, path = classifier.predict_single_with_path(single_instance)
    
    print(f"Instance values: {dict(single_instance)}")
    print(f"Prediction: {prediction}")
    print(f"Actual: {y_series.iloc[0]}")
    
    print_decision_path(path, X_df.columns.tolist(), "Single Instance Decision Path")


if __name__ == "__main__":
    print("GATree Path Interpretation Examples")
    print("This demonstrates how to interpret decision paths in GATree models.")
    print()
    
    try:
        classification_example()
        regression_example()
        action_selection_example()
        continuous_action_example()
        single_prediction_example()
        
        print("=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()