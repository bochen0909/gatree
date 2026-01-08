"""
Debug script to understand why continuous action selector produces limited action diversity.
"""

import numpy as np
import pandas as pd
from gatree.methods.gatreecontinuousactionselector import GATreeContinuousActionSelector


def simple_reward(state, action, y_data, timestep, previous_action):
    """Simple reward function that should encourage diverse actions."""
    # Reward is higher when action matches the 'trend' feature
    target_action = float(state['trend'])  # Use trend as target action
    
    # Reward based on how close action is to target
    distance = abs(action - target_action)
    reward = 1.0 - distance  # Higher reward for closer actions
    
    return reward


def main():
    print("Debugging Continuous Action Diversity")
    print("=" * 40)
    
    # Create simple test data
    np.random.seed(42)
    n_timesteps = 50
    
    # Create features where trend varies from 0 to 1
    trend = np.linspace(0.1, 0.9, n_timesteps)  # Linearly increasing trend
    volatility = np.full(n_timesteps, 0.2)      # Constant volatility
    sentiment = np.full(n_timesteps, 0.5)       # Constant sentiment
    
    X = pd.DataFrame({
        'trend': trend,
        'volatility': volatility,
        'sentiment': sentiment
    })
    
    Y = pd.DataFrame({
        'dummy': np.zeros(n_timesteps)  # Not used in reward function
    })
    
    print(f"Data shape: X={X.shape}")
    print(f"Trend range: {trend.min():.3f} to {trend.max():.3f}")
    
    # Create selector with different configurations
    configs = [
        {"n_action_bins": 10, "max_depth": 3, "name": "10 bins, depth 3"},
        {"n_action_bins": 20, "max_depth": 5, "name": "20 bins, depth 5"},
        {"n_action_bins": 50, "max_depth": 8, "name": "50 bins, depth 8"},
    ]
    
    for config in configs:
        print(f"\n--- Testing: {config['name']} ---")
        
        selector = GATreeContinuousActionSelector(
            reward_function=simple_reward,
            action_bounds=(0, 1),
            n_action_bins=config['n_action_bins'],
            max_depth=config['max_depth'],
            random_state=42
        )
        
        # Train with small population for quick test
        selector.fit(
            X, Y,
            population_size=20,
            max_iter=50,
            mutation_probability=0.3,
            elite_size=2
        )
        
        # Make predictions
        actions = selector.predict_actions(X)
        unique_actions = len(set(actions))
        
        print(f"  Tree size: {selector._tree.size()}")
        print(f"  Tree depth: {selector._tree.max_depth()}")
        print(f"  Unique actions: {unique_actions}")
        print(f"  Action range: {min(actions):.3f} to {max(actions):.3f}")
        print(f"  Action std: {np.std(actions):.3f}")
        
        # Show first few predictions vs targets
        print("  First 10 predictions vs targets:")
        for i in range(min(10, len(actions))):
            print(f"    Step {i:2d}: action={actions[i]:.3f}, target={trend[i]:.3f}, diff={abs(actions[i]-trend[i]):.3f}")
    
    print("\n--- Manual Tree Analysis ---")
    # Let's manually examine what the tree looks like
    selector = GATreeContinuousActionSelector(
        reward_function=simple_reward,
        action_bounds=(0, 1),
        n_action_bins=20,
        max_depth=5,
        random_state=42
    )
    
    selector.fit(X, Y, population_size=30, max_iter=100)
    
    def analyze_tree_node(node, depth=0, path="root"):
        """Recursively analyze tree structure."""
        indent = "  " * depth
        
        if node.att_index == -1:  # Leaf node
            continuous_action = selector._action_index_to_continuous(node.att_value)
            print(f"{indent}{path}: LEAF -> action_idx={node.att_value}, continuous={continuous_action:.3f}")
        else:
            print(f"{indent}{path}: SPLIT on feature_{node.att_index} <= {node.att_value:.3f}")
            if hasattr(node, 'left') and node.left:
                analyze_tree_node(node.left, depth+1, f"{path}/left")
            if hasattr(node, 'right') and node.right:
                analyze_tree_node(node.right, depth+1, f"{path}/right")
    
    print("Tree structure:")
    analyze_tree_node(selector._tree)
    
    # Test predictions on specific values
    print("\nTesting specific predictions:")
    test_cases = [
        {'trend': 0.1, 'volatility': 0.2, 'sentiment': 0.5},
        {'trend': 0.5, 'volatility': 0.2, 'sentiment': 0.5},
        {'trend': 0.9, 'volatility': 0.2, 'sentiment': 0.5},
    ]
    
    for i, case in enumerate(test_cases):
        test_state = pd.Series(case)
        predicted_action = selector.predict_action(test_state)
        print(f"  Case {i+1}: trend={case['trend']:.1f} -> action={predicted_action:.3f}")


if __name__ == "__main__":
    main()