#!/usr/bin/env python3
"""
Simple test for early stopping functionality
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from gatree.methods.gatreeclassifier import GATreeClassifier


def simple_progress_callback(generation, best_fitness, avg_fitness):
    """Simple progress callback"""
    print(f"Gen {generation}: Best={best_fitness:.4f}, Avg={avg_fitness:.4f}")


def test_early_stopping():
    """Test early stopping with a simple example"""
    print("Testing Early Stopping with GATreeClassifier")
    print("=" * 50)
    
    # Generate simple data
    X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y = pd.Series(y)
    
    print("Training with early stopping enabled...")
    classifier = GATreeClassifier(random_state=42)
    
    classifier.fit(
        X, y,
        population_size=10,
        max_iter=30,
        early_stopping=True,
        patience=5,
        min_delta=0.01,
        restore_best_weights=True,
        progress_callback=simple_progress_callback
    )
    
    print(f"\nTraining completed!")
    print(f"Generations run: {len(classifier._best_fitness)}")
    print(f"Best fitness achieved: {min(classifier._best_fitness):.4f}")
    
    # Test predictions
    predictions = classifier.predict(X)
    accuracy = np.mean(predictions == y)
    print(f"Training accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    test_early_stopping()