"""
Example demonstrating sample weight support in GATree models.

This example shows how to use sample weights with both GATreeRegressor
and GATreeActionSelector to give different importance to different samples.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# GATree imports
from gatree.methods.gatreeregressor import GATreeRegressor
from gatree.methods.gatreeactionselector import GATreeActionSelector


def test_regressor_with_weights():
    """Test GATreeRegressor with sample weights."""
    print("Testing GATreeRegressor with sample weights...")
    
    # Generate synthetic regression data
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y = pd.Series(y)
    
    # Create sample weights - give more importance to first half of samples
    sample_weight = np.ones(len(X))
    sample_weight[:50] = 2.0  # Double weight for first 50 samples
    
    # Split data
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, sample_weight, test_size=0.3, random_state=42
    )
    
    # Train without weights
    regressor_no_weights = GATreeRegressor(random_state=42)
    regressor_no_weights.fit(X_train, y_train, max_iter=50)
    
    # Train with weights
    regressor_with_weights = GATreeRegressor(random_state=42)
    regressor_with_weights.fit(X_train, y_train, sample_weight=w_train, max_iter=50)
    
    # Make predictions
    y_pred_no_weights = regressor_no_weights.predict(X_test)
    y_pred_with_weights = regressor_with_weights.predict(X_test)
    
    # Calculate MSE
    mse_no_weights = mean_squared_error(y_test, y_pred_no_weights)
    mse_with_weights = mean_squared_error(y_test, y_pred_with_weights)
    
    print(f"MSE without weights: {mse_no_weights:.4f}")
    print(f"MSE with weights: {mse_with_weights:.4f}")
    print(f"Difference: {abs(mse_no_weights - mse_with_weights):.4f}")
    print()


def test_action_selector_with_weights():
    """Test GATreeActionSelector with sample weights."""
    print("Testing GATreeActionSelector with sample weights...")
    
    # Generate synthetic time series data
    np.random.seed(42)
    n_timesteps = 50
    n_features = 3
    
    # Create features (price, volume, momentum indicators)
    X = pd.DataFrame({
        'price': np.cumsum(np.random.randn(n_timesteps) * 0.1) + 100,
        'volume': np.random.exponential(1000, n_timesteps),
        'momentum': np.random.randn(n_timesteps)
    })
    
    # Create reward calculation data (next period returns)
    Y = pd.DataFrame({
        'next_return': np.random.randn(n_timesteps) * 0.02
    })
    
    # Define action space and reward function
    action_space = ['buy', 'sell', 'hold']
    
    def simple_reward_function(state, action, y_data, timestep):
        """Simple reward based on action and next return."""
        next_return = y_data['next_return']
        if action == 'buy':
            return next_return * 100  # Profit from buying
        elif action == 'sell':
            return -next_return * 100  # Profit from selling
        else:  # hold
            return 0
    
    # Create sample weights - give more importance to later timesteps
    sample_weight = np.linspace(0.5, 2.0, n_timesteps)
    
    # Train without weights
    selector_no_weights = GATreeActionSelector(
        action_space=action_space,
        reward_function=simple_reward_function,
        random_state=42
    )
    selector_no_weights.fit(X, Y, max_iter=20)
    
    # Train with weights
    selector_with_weights = GATreeActionSelector(
        action_space=action_space,
        reward_function=simple_reward_function,
        random_state=42
    )
    selector_with_weights.fit(X, Y, sample_weight=sample_weight, max_iter=20)
    
    # Get action predictions
    actions_no_weights = selector_no_weights.predict_actions(X)
    actions_with_weights = selector_with_weights.predict_actions(X)
    
    # Calculate action distributions
    dist_no_weights = selector_no_weights.get_action_distribution(X)
    dist_with_weights = selector_with_weights.get_action_distribution(X)
    
    print("Action distribution without weights:")
    for action, count in dist_no_weights.items():
        print(f"  {action}: {count} ({count/len(X)*100:.1f}%)")
    
    print("\nAction distribution with weights:")
    for action, count in dist_with_weights.items():
        print(f"  {action}: {count} ({count/len(X)*100:.1f}%)")
    
    # Calculate total rewards
    total_reward_no_weights, _, _ = selector_no_weights.simulate_rewards(X, Y)
    total_reward_with_weights, _, _ = selector_with_weights.simulate_rewards(X, Y)
    
    print(f"\nTotal reward without weights: {total_reward_no_weights:.2f}")
    print(f"Total reward with weights: {total_reward_with_weights:.2f}")
    print(f"Difference: {abs(total_reward_no_weights - total_reward_with_weights):.2f}")
    print()


def test_weight_validation():
    """Test sample weight validation."""
    print("Testing sample weight validation...")
    
    # Create simple data
    X = pd.DataFrame({'feature': [1, 2, 3, 4, 5]})
    y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    
    regressor = GATreeRegressor()
    
    # Test with correct weights
    try:
        weights = np.array([1.0, 1.0, 2.0, 1.0, 1.0])
        regressor.fit(X, y, sample_weight=weights, max_iter=5)
        print("✓ Correct weights accepted")
    except Exception as e:
        print(f"✗ Unexpected error with correct weights: {e}")
    
    # Test with wrong length
    try:
        weights = np.array([1.0, 1.0, 2.0])  # Wrong length
        regressor.fit(X, y, sample_weight=weights, max_iter=5)
        print("✗ Wrong length weights should have failed")
    except ValueError as e:
        print(f"✓ Wrong length weights correctly rejected: {e}")
    
    # Test with negative weights
    try:
        weights = np.array([1.0, -1.0, 2.0, 1.0, 1.0])  # Negative weight
        regressor.fit(X, y, sample_weight=weights, max_iter=5)
        print("✗ Negative weights should have failed")
    except ValueError as e:
        print(f"✓ Negative weights correctly rejected: {e}")
    
    print()


if __name__ == "__main__":
    print("Sample Weight Support Example")
    print("=" * 40)
    
    test_weight_validation()
    test_regressor_with_weights()
    test_action_selector_with_weights()
    
    print("Sample weight functionality implemented successfully!")