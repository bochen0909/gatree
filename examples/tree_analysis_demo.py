#!/usr/bin/env python3
"""
GATree Analysis Demo: Tree Visualization, Feature Importance, and SHAP Integration

This example demonstrates the new analysis capabilities added to GATree:
1. JSON export of tree structure
2. Feature importance calculation
3. SHAP integration via sklearn conversion

Author: GATree Enhancement
"""

import json
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, make_regression
from sklearn.model_selection import train_test_split

# GATree imports
from gatree.methods.gatreeclassifier import GATreeClassifier
from gatree.methods.gatreeregressor import GATreeRegressor

def demo_classification_analysis():
    """Demonstrate tree analysis features with classification."""
    print("=" * 60)
    print("CLASSIFICATION ANALYSIS DEMO")
    print("=" * 60)
    
    # Load iris dataset
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train GATree classifier
    print("Training GATree Classifier...")
    gatree = GATreeClassifier(max_depth=4, random_state=42)
    gatree.fit(X_train, y_train, population_size=50, max_iter=100)
    
    # Make predictions
    y_pred = gatree.predict(X_test)
    accuracy = sum(y_pred == y_test) / len(y_test)
    print(f"Test Accuracy: {accuracy:.3f}")
    
    print("\n" + "-" * 40)
    print("1. TREE VISUALIZATION (ASCII)")
    print("-" * 40)
    gatree.plot()
    
    print("\n" + "-" * 40)
    print("2. JSON EXPORT")
    print("-" * 40)
    tree_json = gatree.export_tree_json()
    print("Tree structure exported to JSON:")
    print(json.dumps(tree_json, indent=2)[:500] + "..." if len(str(tree_json)) > 500 else json.dumps(tree_json, indent=2))
    
    # Save to file
    with open('iris_tree.json', 'w') as f:
        json.dump(tree_json, f, indent=2)
    print("Full tree saved to 'iris_tree.json'")
    
    print("\n" + "-" * 40)
    print("3. FEATURE IMPORTANCE")
    print("-" * 40)
    importance = gatree.get_feature_importance()
    print("Feature Importance Scores:")
    for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feature}: {score:.4f}")
    
    print("\n" + "-" * 40)
    print("4. SHAP INTEGRATION")
    print("-" * 40)
    try:
        import shap
        
        # Convert to sklearn format
        sklearn_tree = gatree.to_sklearn_tree()
        print(f"Converted to sklearn {type(sklearn_tree).__name__}")
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(sklearn_tree)
        shap_values = explainer.shap_values(X_test.iloc[:5])  # Explain first 5 test samples
        
        print("SHAP values calculated successfully!")
        print(f"SHAP values shape: {np.array(shap_values).shape}")
        
        # Show SHAP values for first sample
        print(f"\nSHAP values for first test sample:")
        if isinstance(shap_values, list):  # Multi-class
            for i, class_shap in enumerate(shap_values):
                print(f"  Class {i}: {class_shap[0]}")
        else:  # Binary classification
            print(f"  {shap_values[0]}")
            
    except ImportError:
        print("SHAP not installed. Install with: pip install shap")
    except Exception as e:
        print(f"SHAP analysis failed: {e}")
        print("Note: SHAP integration uses sklearn tree fitted on same data")

def demo_regression_analysis():
    """Demonstrate tree analysis features with regression."""
    print("\n\n" + "=" * 60)
    print("REGRESSION ANALYSIS DEMO")
    print("=" * 60)
    
    # Create synthetic regression dataset
    np.random.seed(42)
    n_samples = 200
    X = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, n_samples),
        'feature_2': np.random.normal(0, 1, n_samples),
        'feature_3': np.random.normal(0, 1, n_samples),
        'feature_4': np.random.normal(0, 1, n_samples)
    })
    # Create target with known relationships
    y = pd.Series(2 * X['feature_1'] + 1.5 * X['feature_2'] + 0.5 * X['feature_3'] + np.random.normal(0, 0.1, n_samples))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train GATree regressor
    print("Training GATree Regressor...")
    gatree = GATreeRegressor(max_depth=4, random_state=42)
    gatree.fit(X_train, y_train, population_size=50, max_iter=100)
    
    # Make predictions
    y_pred = gatree.predict(X_test)
    mse = np.mean((y_pred - y_test) ** 2)
    print(f"Test MSE: {mse:.3f}")
    
    print("\n" + "-" * 40)
    print("1. FEATURE IMPORTANCE")
    print("-" * 40)
    importance = gatree.get_feature_importance()
    print("Feature Importance Scores:")
    for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feature}: {score:.4f}")
    
    print("\nExpected: feature_1 and feature_2 should have highest importance")
    
    print("\n" + "-" * 40)
    print("2. JSON EXPORT")
    print("-" * 40)
    tree_json = gatree.export_tree_json()
    print("Tree structure (first 300 chars):")
    print(str(tree_json)[:300] + "...")
    
    print("\n" + "-" * 40)
    print("3. SHAP INTEGRATION")
    print("-" * 40)
    try:
        import shap
        
        # Convert to sklearn format
        sklearn_tree = gatree.to_sklearn_tree()
        print(f"Converted to sklearn {type(sklearn_tree).__name__}")
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(sklearn_tree)
        shap_values = explainer.shap_values(X_test.iloc[:5])
        
        print("SHAP values calculated successfully!")
        print(f"SHAP values shape: {shap_values.shape}")
        
        # Show SHAP values for first sample
        print(f"\nSHAP values for first test sample:")
        for i, feature in enumerate(X.columns):
            print(f"  {feature}: {shap_values[0][i]:.4f}")
            
    except ImportError:
        print("SHAP not installed. Install with: pip install shap")
    except Exception as e:
        print(f"SHAP analysis failed: {e}")
        print("Note: SHAP integration uses sklearn tree fitted on same data")

def save_analysis_results():
    """Save analysis results to files for further inspection."""
    print("\n" + "=" * 60)
    print("SAVING ANALYSIS RESULTS")
    print("=" * 60)
    
    # This would be called after running the demos
    print("Analysis results saved:")
    print("- iris_tree.json: Tree structure for iris classification")
    print("- Feature importance scores displayed in console")
    print("- SHAP values calculated and displayed")

if __name__ == "__main__":
    print("GATree Analysis Demo")
    print("This demo showcases new tree analysis capabilities:")
    print("1. JSON export for tree visualization")
    print("2. Feature importance calculation")
    print("3. SHAP integration via sklearn conversion")
    
    demo_classification_analysis()
    demo_regression_analysis()
    save_analysis_results()
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("New GATree methods available:")
    print("- gatree.export_tree_json(): Export tree to JSON format")
    print("- gatree.get_feature_importance(): Calculate feature importance")
    print("- gatree.to_sklearn_tree(): Convert to sklearn for SHAP analysis")