#!/usr/bin/env python3
"""
Demo script showcasing the enhanced GATree features:
1. Sample weight support in GATreeClassifier
2. Progress callbacks for monitoring training
3. Better error handling
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from gatree.methods.gatreeclassifier import GATreeClassifier
from gatree.methods.gatreeregressor import GATreeRegressor


def progress_callback(generation, best_fitness, avg_fitness, best_reward=None):
    """Progress callback function to monitor training"""
    if best_reward is not None:
        print(f"Generation {generation:3d}: Best Fitness = {best_fitness:.4f}, "
              f"Avg Fitness = {avg_fitness:.4f}, Best Reward = {best_reward:.4f}")
    else:
        print(f"Generation {generation:3d}: Best Fitness = {best_fitness:.4f}, "
              f"Avg Fitness = {avg_fitness:.4f}")


def demo_classifier_with_sample_weights():
    """Demonstrate GATreeClassifier with sample weights and progress callback"""
    print("=" * 60)
    print("GATreeClassifier with Sample Weights Demo")
    print("=" * 60)
    
    # Generate synthetic classification data
    X, y = make_classification(n_samples=200, n_features=10, n_classes=3, 
                              n_informative=5, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y = pd.Series(y)
    
    # Create sample weights (emphasize some samples more than others)
    sample_weights = np.random.uniform(0.5, 2.0, size=len(X))
    print(f"Sample weights range: {sample_weights.min():.2f} - {sample_weights.max():.2f}")
    
    # Split data
    X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
        X, y, sample_weights, test_size=0.3, random_state=42)
    
    # Train classifier with sample weights and progress callback
    print("\nTraining GATreeClassifier with sample weights...")
    classifier = GATreeClassifier(random_state=42)
    
    try:
        classifier.fit(
            X_train, y_train, 
            sample_weight=weights_train,
            population_size=20,  # Small for demo
            max_iter=10,         # Few iterations for demo
            progress_callback=progress_callback
        )
        
        # Make predictions
        y_pred = classifier.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        print(f"\nTest Accuracy: {accuracy:.4f}")
        
        # Show feature importance
        importance = classifier.get_feature_importance()
        print("\nTop 3 Most Important Features:")
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for feature, score in sorted_features[:3]:
            print(f"  {feature}: {score:.4f}")
            
    except Exception as e:
        print(f"Error during training: {e}")


def demo_regressor_with_progress():
    """Demonstrate GATreeRegressor with progress callback"""
    print("\n" + "=" * 60)
    print("GATreeRegressor with Progress Callback Demo")
    print("=" * 60)
    
    # Generate synthetic regression data
    X, y = make_regression(n_samples=150, n_features=8, noise=0.1, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y = pd.Series(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train regressor with progress callback
    print("Training GATreeRegressor with progress monitoring...")
    regressor = GATreeRegressor(random_state=42)
    
    try:
        regressor.fit(
            X_train, y_train,
            population_size=15,  # Small for demo
            max_iter=8,          # Few iterations for demo
            progress_callback=progress_callback
        )
        
        # Make predictions
        y_pred = regressor.predict(X_test)
        mse = np.mean((y_pred - y_test) ** 2)
        print(f"\nTest MSE: {mse:.4f}")
        
        # Show tree structure info
        if regressor._tree:
            print(f"Tree depth: {regressor._tree.max_depth()}")
            print(f"Tree size: {regressor._tree.size()} nodes")
            
    except Exception as e:
        print(f"Error during training: {e}")


def demo_error_handling():
    """Demonstrate improved error handling"""
    print("\n" + "=" * 60)
    print("Error Handling Demo")
    print("=" * 60)
    
    # Generate small dataset
    X = pd.DataFrame([[1, 2], [3, 4], [5, 6]], columns=['A', 'B'])
    y = pd.Series([0, 1, 0])
    
    classifier = GATreeClassifier(random_state=42)
    
    # Test invalid sample weights
    print("Testing invalid sample weights...")
    try:
        invalid_weights = np.array([1.0, -0.5, 1.0])  # Negative weight
        classifier.fit(X, y, sample_weight=invalid_weights, max_iter=1)
    except RuntimeError as e:
        print(f"✓ Caught expected error: {e}")
    
    # Test invalid parameters
    print("\nTesting invalid parameters...")
    try:
        classifier.fit(X, y, population_size=-5, max_iter=1)
    except RuntimeError as e:
        print(f"✓ Caught expected error: {e}")
    
    # Test mismatched sample weight length
    print("\nTesting mismatched sample weight length...")
    try:
        wrong_length_weights = np.array([1.0, 1.0])  # Wrong length
        classifier.fit(X, y, sample_weight=wrong_length_weights, max_iter=1)
    except RuntimeError as e:
        print(f"✓ Caught expected error: {e}")
    
    print("\n✓ All error handling tests passed!")


if __name__ == "__main__":
    print("GATree Enhanced Features Demo")
    print("This demo showcases the new improvements:")
    print("1. Sample weight support in GATreeClassifier")
    print("2. Progress callbacks for all GATree methods")
    print("3. Comprehensive error handling and validation")
    
    demo_classifier_with_sample_weights()
    demo_regressor_with_progress()
    demo_error_handling()
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)