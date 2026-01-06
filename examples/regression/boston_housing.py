"""
Example of using GATreeRegressor for regression on the Boston Housing dataset.
This example demonstrates how to use the evolutionary decision tree regressor
for predicting continuous target values.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Import the GATreeRegressor
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from gatree.methods.gatreeregressor import GATreeRegressor


def main():
    """
    Main function to demonstrate GATreeRegressor usage.
    """
    print("GATree Regressor Example - California Housing Dataset")
    print("=" * 55)
    
    # Load the California Housing dataset (replacement for deprecated Boston Housing)
    print("Loading California Housing dataset...")
    housing = fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = pd.Series(housing.target, name='target')
    
    print(f"Dataset shape: {X.shape}")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
    print(f"Target mean: {y.mean():.2f}, std: {y.std():.2f}")
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Optional: Scale features for better performance
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), 
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), 
        columns=X_test.columns,
        index=X_test.index
    )
    
    # Create and fit the GATree regressor
    print("\nTraining GATree Regressor...")
    print("Parameters: population_size=50, max_iter=100, max_depth=10")
    
    gatree = GATreeRegressor(
        max_depth=10, 
        n_jobs=4,  # Use 4 cores for parallel processing
        random_state=42
    )
    
    # Fit the model with smaller parameters for faster execution
    gatree.fit(
        X=X_train_scaled, 
        y=y_train, 
        population_size=50,  # Smaller population for faster training
        max_iter=100,        # Fewer iterations for demo
        mutation_probability=0.15,
        elite_size=2,
        selection_tournament_size=3
    )
    
    # Make predictions on the testing set
    print("Making predictions...")
    y_pred = gatree.predict(X_test_scaled)
    
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
    
    # Plot fitness evolution if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(gatree._best_fitness, label='Best Fitness', color='blue')
        plt.plot(gatree._avg_fitness, label='Average Fitness', color='red', alpha=0.7)
        plt.xlabel('Generation')
        plt.ylabel('Fitness (lower is better)')
        plt.title('Fitness Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.scatter(y_test, y_pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Predictions vs Actual (R² = {r2:.3f})')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("\nMatplotlib not available - skipping plots")
    
    # Display tree information
    print(f"\nFinal Tree Information:")
    print(f"Tree depth: {gatree._tree.max_depth()}")
    print(f"Tree size (nodes): {gatree._tree.size()}")
    print(f"Number of leaves: {len(gatree._tree.get_leaves())}")


if __name__ == "__main__":
    main()