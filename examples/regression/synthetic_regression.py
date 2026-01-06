"""
Example of using GATreeRegressor for regression on synthetic data.
This example demonstrates the basic usage of the evolutionary decision tree regressor
with a simple synthetic dataset.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Import the GATreeRegressor
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from gatree.methods.gatreeregressor import GATreeRegressor


def main():
    """
    Main function to demonstrate GATreeRegressor usage with synthetic data.
    """
    print("GATree Regressor Example - Synthetic Regression Dataset")
    print("=" * 60)
    
    # Generate synthetic regression dataset
    print("Generating synthetic regression dataset...")
    X, y = make_regression(
        n_samples=1000,
        n_features=5,
        n_informative=3,
        noise=0.1,
        random_state=42
    )
    
    # Convert to pandas for consistency with GATree interface
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    X = pd.DataFrame(X, columns=feature_names)
    y = pd.Series(y, name='target')
    
    print(f"Dataset shape: {X.shape}")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
    print(f"Target mean: {y.mean():.2f}, std: {y.std():.2f}")
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Create and fit the GATree regressor
    print("\nTraining GATree Regressor...")
    print("Parameters: population_size=30, max_iter=50, max_depth=8")
    
    gatree = GATreeRegressor(
        max_depth=8, 
        n_jobs=2,  # Use 2 cores for parallel processing
        random_state=42
    )
    
    # Fit the model with small parameters for quick demonstration
    gatree.fit(
        X=X_train, 
        y=y_train, 
        population_size=30,   # Small population for quick training
        max_iter=50,          # Few iterations for demo
        mutation_probability=0.2,
        elite_size=1,
        selection_tournament_size=2
    )
    
    # Make predictions on the testing set
    print("Making predictions...")
    y_pred = gatree.predict(X_test)
    
    # Evaluate the regressor
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("\nResults:")
    print("=" * 30)
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Show some example predictions
    print("\nSample Predictions:")
    print("-" * 30)
    print("Actual\tPredicted\tDifference")
    for i in range(min(10, len(y_test))):
        actual = y_test.iloc[i]
        predicted = y_pred[i]
        diff = abs(actual - predicted)
        print(f"{actual:.2f}\t{predicted:.2f}\t\t{diff:.2f}")
    
    # Display fitness evolution
    print(f"\nFitness Evolution:")
    print(f"Initial best fitness: {gatree._best_fitness[0]:.4f}")
    print(f"Final best fitness: {gatree._best_fitness[-1]:.4f}")
    print(f"Improvement: {gatree._best_fitness[0] - gatree._best_fitness[-1]:.4f}")
    
    # Display tree information
    print(f"\nFinal Tree Information:")
    print(f"Tree depth: {gatree._tree.max_depth()}")
    print(f"Tree size (nodes): {gatree._tree.size()}")
    print(f"Number of leaves: {len(gatree._tree.get_leaves())}")
    
    # Show tree structure (first few levels)
    print(f"\nTree Structure (root level):")
    if gatree._tree.att_index != -1:
        print(f"Root split: feature_{gatree._tree.att_index} > {gatree._tree.att_value:.3f}")
        if gatree._tree.left and gatree._tree.left.att_index != -1:
            print(f"  Left child: feature_{gatree._tree.left.att_index} > {gatree._tree.left.att_value:.3f}")
        elif gatree._tree.left:
            print(f"  Left child: leaf with value {gatree._tree.left.att_value:.3f}")
        
        if gatree._tree.right and gatree._tree.right.att_index != -1:
            print(f"  Right child: feature_{gatree._tree.right.att_index} > {gatree._tree.right.att_value:.3f}")
        elif gatree._tree.right:
            print(f"  Right child: leaf with value {gatree._tree.right.att_value:.3f}")
    else:
        print(f"Root is a leaf with value: {gatree._tree.att_value:.3f}")


if __name__ == "__main__":
    main()