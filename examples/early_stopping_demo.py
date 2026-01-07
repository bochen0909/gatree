#!/usr/bin/env python3
"""
Demo script showcasing early stopping functionality in GATree methods:
1. GATreeClassifier with early stopping
2. GATreeRegressor with early stopping  
3. GATreeActionSelector with early stopping
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from gatree.methods.gatreeclassifier import GATreeClassifier
from gatree.methods.gatreeregressor import GATreeRegressor
from gatree.methods.gatreeactionselector import GATreeActionSelector


def progress_callback_with_early_stopping(generation, best_fitness, avg_fitness, best_reward=None):
    """Progress callback function to monitor training with early stopping info"""
    if best_reward is not None:
        print(f"Generation {generation:3d}: Best Fitness = {best_fitness:.4f}, "
              f"Avg Fitness = {avg_fitness:.4f}, Best Reward = {best_reward:.4f}")
    else:
        print(f"Generation {generation:3d}: Best Fitness = {best_fitness:.4f}, "
              f"Avg Fitness = {avg_fitness:.4f}")


def demo_classifier_early_stopping():
    """Demonstrate GATreeClassifier with early stopping"""
    print("=" * 70)
    print("GATreeClassifier with Early Stopping Demo")
    print("=" * 70)
    
    # Generate synthetic classification data
    X, y = make_classification(n_samples=300, n_features=10, n_classes=3, 
                              n_informative=5, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y = pd.Series(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print("Training GATreeClassifier with early stopping...")
    print("Parameters: patience=10, min_delta=0.001")
    
    classifier = GATreeClassifier(random_state=42)
    
    try:
        classifier.fit(
            X_train, y_train,
            population_size=30,
            max_iter=100,  # High max_iter to test early stopping
            early_stopping=True,
            patience=10,
            min_delta=0.001,
            restore_best_weights=True,
            progress_callback=progress_callback_with_early_stopping
        )
        
        # Make predictions
        y_pred = classifier.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        print(f"\nTest Accuracy: {accuracy:.4f}")
        print(f"Total generations run: {len(classifier._best_fitness)}")
        print(f"Final best fitness: {classifier._best_fitness[-1]:.4f}")
        
    except Exception as e:
        print(f"Error during training: {e}")


def demo_regressor_early_stopping():
    """Demonstrate GATreeRegressor with early stopping"""
    print("\n" + "=" * 70)
    print("GATreeRegressor with Early Stopping Demo")
    print("=" * 70)
    
    # Generate synthetic regression data
    X, y = make_regression(n_samples=200, n_features=8, noise=0.1, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y = pd.Series(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print("Training GATreeRegressor with early stopping...")
    print("Parameters: patience=15, min_delta=0.01")
    
    regressor = GATreeRegressor(random_state=42)
    
    try:
        regressor.fit(
            X_train, y_train,
            population_size=25,
            max_iter=80,  # High max_iter to test early stopping
            early_stopping=True,
            patience=15,
            min_delta=0.01,
            restore_best_weights=True,
            progress_callback=progress_callback_with_early_stopping
        )
        
        # Make predictions
        y_pred = regressor.predict(X_test)
        mse = np.mean((y_pred - y_test) ** 2)
        print(f"\nTest MSE: {mse:.4f}")
        print(f"Total generations run: {len(regressor._best_fitness)}")
        print(f"Final best fitness: {regressor._best_fitness[-1]:.4f}")
        
    except Exception as e:
        print(f"Error during training: {e}")


def demo_action_selector_early_stopping():
    """Demonstrate GATreeActionSelector with early stopping"""
    print("\n" + "=" * 70)
    print("GATreeActionSelector with Early Stopping Demo")
    print("=" * 70)
    
    # Create simple time series data
    np.random.seed(42)
    n_timesteps = 50
    n_features = 4
    
    # Generate features (market conditions)
    X = pd.DataFrame({
        'price': np.cumsum(np.random.randn(n_timesteps) * 0.1) + 100,
        'volume': np.random.exponential(1000, n_timesteps),
        'volatility': np.abs(np.random.randn(n_timesteps)),
        'trend': np.sin(np.arange(n_timesteps) * 0.1)
    })
    
    # Y contains price data for reward calculation
    Y = pd.DataFrame({
        'price': X['price'].values,
        'next_price': np.roll(X['price'].values, -1)  # Next period price
    })
    Y.loc[len(Y)-1, 'next_price'] = Y.loc[len(Y)-1, 'price']  # Handle last timestep
    
    # Define action space and reward function
    action_space = ['buy', 'sell', 'hold']
    
    def simple_reward_function(state, action, y_data, timestep, previous_action):
        """Simple reward function based on price movement prediction"""
        current_price = y_data['price']
        next_price = y_data['next_price']
        price_change = next_price - current_price
        
        if action == 'buy' and price_change > 0:
            return price_change  # Profit from buying before price increase
        elif action == 'sell' and price_change < 0:
            return -price_change  # Profit from selling before price decrease
        elif action == 'hold':
            return 0.1  # Small reward for holding
        else:
            return -abs(price_change) * 0.5  # Penalty for wrong prediction
    
    print("Training GATreeActionSelector with early stopping...")
    print("Parameters: patience=8, min_delta=0.1")
    
    action_selector = GATreeActionSelector(
        action_space=action_space,
        reward_function=simple_reward_function,
        random_state=42
    )
    
    try:
        action_selector.fit(
            X, Y,
            population_size=20,
            max_iter=60,  # High max_iter to test early stopping
            early_stopping=True,
            patience=8,
            min_delta=0.1,
            restore_best_weights=True,
            progress_callback=progress_callback_with_early_stopping
        )
        
        # Make predictions
        actions = action_selector.predict_actions(X)
        total_reward, individual_rewards, _ = action_selector.simulate_rewards(X, Y)
        
        print(f"\nTotal Reward: {total_reward:.4f}")
        print(f"Average Reward per Timestep: {np.mean(individual_rewards):.4f}")
        print(f"Total generations run: {len(action_selector._best_fitness)}")
        print(f"Final best fitness: {action_selector._best_fitness[-1]:.4f}")
        
        # Show action distribution
        action_dist = action_selector.get_action_distribution(X)
        print(f"Action Distribution: {action_dist}")
        
    except Exception as e:
        print(f"Error during training: {e}")


def demo_comparison_with_without_early_stopping():
    """Compare training with and without early stopping"""
    print("\n" + "=" * 70)
    print("Comparison: With vs Without Early Stopping")
    print("=" * 70)
    
    # Generate simple classification data
    X, y = make_classification(n_samples=150, n_features=5, n_classes=2, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y = pd.Series(y)
    
    print("Training WITHOUT early stopping (50 generations)...")
    classifier_no_early = GATreeClassifier(random_state=42)
    classifier_no_early.fit(X, y, population_size=20, max_iter=50)
    
    print(f"Generations run: {len(classifier_no_early._best_fitness)}")
    print(f"Final fitness: {classifier_no_early._best_fitness[-1]:.4f}")
    
    print("\nTraining WITH early stopping (max 50 generations, patience=5)...")
    classifier_early = GATreeClassifier(random_state=42)
    classifier_early.fit(
        X, y, 
        population_size=20, 
        max_iter=50,
        early_stopping=True,
        patience=5,
        min_delta=0.001,
        progress_callback=progress_callback_with_early_stopping
    )
    
    print(f"Generations run: {len(classifier_early._best_fitness)}")
    print(f"Final fitness: {classifier_early._best_fitness[-1]:.4f}")
    
    # Compare efficiency
    efficiency_gain = (len(classifier_no_early._best_fitness) - len(classifier_early._best_fitness)) / len(classifier_no_early._best_fitness) * 100
    print(f"\nEfficiency gain: {efficiency_gain:.1f}% fewer generations with early stopping")


if __name__ == "__main__":
    print("GATree Early Stopping Demo")
    print("This demo showcases early stopping functionality for all GATree methods:")
    print("1. GATreeClassifier with early stopping")
    print("2. GATreeRegressor with early stopping")
    print("3. GATreeActionSelector with early stopping")
    print("4. Comparison with and without early stopping")
    
    demo_classifier_early_stopping()
    demo_regressor_early_stopping()
    demo_action_selector_early_stopping()
    demo_comparison_with_without_early_stopping()
    
    print("\n" + "=" * 70)
    print("Early Stopping Demo completed successfully!")
    print("=" * 70)
    print("\nKey Early Stopping Parameters:")
    print("- early_stopping: Enable/disable early stopping (default: False)")
    print("- patience: Number of generations to wait for improvement (default: 50)")
    print("- min_delta: Minimum improvement threshold (default: 1e-4)")
    print("- restore_best_weights: Restore best tree when stopping (default: True)")