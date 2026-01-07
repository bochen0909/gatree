#!/usr/bin/env python3
"""
Comprehensive test for early stopping in all GATree methods
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

from gatree.methods.gatreeclassifier import GATreeClassifier
from gatree.methods.gatreeregressor import GATreeRegressor
from gatree.methods.gatreeactionselector import GATreeActionSelector


def test_classifier_early_stopping():
    """Test GATreeClassifier early stopping"""
    print("Testing GATreeClassifier Early Stopping...")
    
    X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
    X = pd.DataFrame(X)
    y = pd.Series(y)
    
    classifier = GATreeClassifier(random_state=42)
    classifier.fit(
        X, y,
        population_size=10,
        max_iter=50,
        early_stopping=True,
        patience=5,
        min_delta=0.01
    )
    
    print(f"✓ Classifier: {len(classifier._best_fitness)} generations (max 50)")
    return len(classifier._best_fitness) < 50  # Should stop early


def test_regressor_early_stopping():
    """Test GATreeRegressor early stopping"""
    print("Testing GATreeRegressor Early Stopping...")
    
    X, y = make_regression(n_samples=100, n_features=5, random_state=42)
    X = pd.DataFrame(X)
    y = pd.Series(y)
    
    regressor = GATreeRegressor(random_state=42)
    regressor.fit(
        X, y,
        population_size=10,
        max_iter=50,
        early_stopping=True,
        patience=5,
        min_delta=0.01
    )
    
    print(f"✓ Regressor: {len(regressor._best_fitness)} generations (max 50)")
    return len(regressor._best_fitness) < 50  # Should stop early


def test_action_selector_early_stopping():
    """Test GATreeActionSelector early stopping"""
    print("Testing GATreeActionSelector Early Stopping...")
    
    # Simple time series data
    X = pd.DataFrame({
        'feature1': np.random.randn(20),
        'feature2': np.random.randn(20)
    })
    Y = pd.DataFrame({
        'reward_data': np.random.randn(20)
    })
    
    def simple_reward(state, action, y_data, timestep, previous_action):
        return np.random.randn()  # Random reward for testing
    
    action_selector = GATreeActionSelector(
        action_space=['A', 'B'],
        reward_function=simple_reward,
        random_state=42
    )
    
    action_selector.fit(
        X, Y,
        population_size=5,
        max_iter=30,
        early_stopping=True,
        patience=3,
        min_delta=0.1
    )
    
    print(f"✓ ActionSelector: {len(action_selector._best_fitness)} generations (max 30)")
    return len(action_selector._best_fitness) < 30  # Should stop early


def test_parameter_validation():
    """Test early stopping parameter validation"""
    print("Testing Parameter Validation...")
    
    X, y = make_classification(n_samples=50, n_features=5, n_informative=3, random_state=42)
    X = pd.DataFrame(X)
    y = pd.Series(y)
    
    classifier = GATreeClassifier(random_state=42)
    
    # Test invalid patience
    try:
        classifier.fit(X, y, early_stopping=True, patience=0, max_iter=1)
        print("✗ Should have failed with patience=0")
        return False
    except (ValueError, RuntimeError) as e:
        if "patience must be positive" in str(e):
            print("✓ Correctly rejected patience=0")
        else:
            print(f"✓ Correctly rejected invalid parameters: {e}")
    
    # Test invalid min_delta
    try:
        classifier.fit(X, y, early_stopping=True, min_delta=-0.1, max_iter=1)
        print("✗ Should have failed with negative min_delta")
        return False
    except (ValueError, RuntimeError) as e:
        if "min_delta must be non-negative" in str(e):
            print("✓ Correctly rejected negative min_delta")
        else:
            print(f"✓ Correctly rejected invalid parameters: {e}")
    
    return True


def main():
    """Run all tests"""
    print("=" * 60)
    print("GATree Early Stopping Comprehensive Test")
    print("=" * 60)
    
    tests = [
        test_classifier_early_stopping,
        test_regressor_early_stopping,
        test_action_selector_early_stopping,
        test_parameter_validation
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print()
        except Exception as e:
            print(f"✗ Test failed with error: {e}")
            results.append(False)
            print()
    
    print("=" * 60)
    print("Test Results:")
    print(f"Passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("🎉 All early stopping tests passed!")
    else:
        print("❌ Some tests failed")
    
    print("=" * 60)


if __name__ == "__main__":
    main()