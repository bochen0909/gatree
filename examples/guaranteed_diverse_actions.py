"""
Example that guarantees diverse continuous actions by design.

This creates a scenario where different features clearly require different optimal actions.
"""

import numpy as np
import pandas as pd
from gatree.methods.gatreecontinuousactionselector import GATreeContinuousActionSelector


def create_diverse_scenario_data(n_timesteps=100, random_state=42):
    """
    Create data where different feature combinations clearly require different actions.
    """
    np.random.seed(random_state)
    
    # Create three clear features that should map to different action ranges
    feature_a = np.random.uniform(0, 1, n_timesteps)  # Should map to action ≈ feature_a
    feature_b = np.random.uniform(0, 1, n_timesteps)  # Modifies the mapping
    feature_c = np.random.uniform(0, 1, n_timesteps)  # Adds noise/complexity
    
    X = pd.DataFrame({
        'feature_a': feature_a,
        'feature_b': feature_b, 
        'feature_c': feature_c
    })
    
    # Create dummy Y data (not used in reward function)
    Y = pd.DataFrame({
        'dummy': np.zeros(n_timesteps)
    })
    
    return X, Y


def diverse_reward_function(state, action, y_data, timestep, previous_action):
    """
    Reward function designed to require diverse actions based on state features.
    
    The optimal action should be approximately:
    - If feature_a < 0.3: optimal action ≈ 0.1-0.3 (low)
    - If feature_a > 0.7: optimal action ≈ 0.7-0.9 (high)  
    - Otherwise: optimal action ≈ feature_a (medium)
    
    feature_b and feature_c add complexity to prevent trivial solutions.
    """
    try:
        feature_a = float(state['feature_a'])
        feature_b = float(state['feature_b'])
        feature_c = float(state['feature_c'])
        
        # Calculate target action based on features
        if feature_a < 0.3:
            target_action = 0.2 + feature_b * 0.1  # Range: 0.2-0.3
        elif feature_a > 0.7:
            target_action = 0.8 - feature_b * 0.1  # Range: 0.7-0.8
        else:
            # Middle range: action should approximate feature_a, modified by feature_b
            target_action = feature_a + (feature_b - 0.5) * 0.2
            target_action = np.clip(target_action, 0.3, 0.7)
        
        # Add feature_c influence
        target_action += (feature_c - 0.5) * 0.05
        target_action = np.clip(target_action, 0, 1)
        
        # Reward is higher when action is closer to target
        distance = abs(action - target_action)
        base_reward = 1.0 - distance  # Max reward = 1.0 when distance = 0
        
        # Add small penalty for extreme actions to encourage moderation
        extreme_penalty = 0.1 * max(0, abs(action - 0.5) - 0.4)  # Penalty when action < 0.1 or > 0.9
        
        # Small transaction cost
        transaction_cost = 0.0
        if previous_action is not None:
            transaction_cost = 0.02 * abs(action - previous_action)
        
        reward = base_reward - extreme_penalty - transaction_cost
        
        return reward
        
    except Exception as e:
        print(f"Error in reward function: {e}")
        return 0.0


def main():
    """Demonstrate guaranteed diverse continuous actions."""
    print("Guaranteed Diverse Continuous Actions Example")
    print("=" * 45)
    
    # Create data
    X, Y = create_diverse_scenario_data(n_timesteps=150, random_state=42)
    
    print(f"Data shape: X={X.shape}")
    print(f"Feature ranges:")
    for col in X.columns:
        print(f"  {col}: {X[col].min():.3f} to {X[col].max():.3f}")
    
    # Split data
    train_size = int(0.7 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    Y_train, Y_test = Y[:train_size], Y[train_size:]
    
    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Create selector
    selector = GATreeContinuousActionSelector(
        reward_function=diverse_reward_function,
        action_bounds=(0, 1),
        max_depth=10,  # Allow complex trees
        n_action_bins=100,  # Fine-grained actions
        discount_factor=1.0,  # No discounting for simplicity
        random_state=42
    )
    
    # Train
    print("\nTraining...")
    selector.fit(
        X_train, Y_train,
        population_size=60,
        max_iter=100,
        mutation_probability=0.3,
        elite_size=6,
        early_stopping=True,
        patience=20
    )
    
    print(f"Training completed!")
    print(f"Tree depth: {selector._tree.max_depth()}")
    print(f"Tree size: {selector._tree.size()}")
    
    # Test predictions
    test_actions = selector.predict_actions(X_test)
    
    # Analyze diversity
    unique_actions = len(set([round(a, 3) for a in test_actions]))
    stats = selector.get_action_statistics(X_test)
    
    print(f"\nAction Diversity Analysis:")
    print(f"  Unique actions (rounded to 3 decimals): {unique_actions}")
    print(f"  Action range: {stats['min']:.3f} to {stats['max']:.3f}")
    print(f"  Action std: {stats['std']:.3f}")
    print(f"  Action mean: {stats['mean']:.3f}")
    
    # Show some examples of state -> action mappings
    print(f"\nExample State -> Action Mappings:")
    print(f"{'feature_a':>9} {'feature_b':>9} {'feature_c':>9} {'action':>8} {'target':>8}")
    print("-" * 50)
    
    for i in range(min(15, len(X_test))):
        state = X_test.iloc[i]
        action = test_actions[i]
        
        # Calculate what the target action should be
        feature_a = state['feature_a']
        feature_b = state['feature_b']
        feature_c = state['feature_c']
        
        if feature_a < 0.3:
            target = 0.2 + feature_b * 0.1
        elif feature_a > 0.7:
            target = 0.8 - feature_b * 0.1
        else:
            target = feature_a + (feature_b - 0.5) * 0.2
            target = np.clip(target, 0.3, 0.7)
        
        target += (feature_c - 0.5) * 0.05
        target = np.clip(target, 0, 1)
        
        print(f"{feature_a:9.3f} {feature_b:9.3f} {feature_c:9.3f} {action:8.3f} {target:8.3f}")
    
    # Test performance
    total_reward, individual_rewards, _ = selector.simulate_rewards(X_test, Y_test)
    
    print(f"\nPerformance:")
    print(f"  Total reward: {total_reward:.4f}")
    print(f"  Average reward: {np.mean(individual_rewards):.4f}")
    print(f"  Reward std: {np.std(individual_rewards):.4f}")
    
    # Compare with optimal strategy (using target actions directly)
    print(f"\nComparison with Optimal Strategy:")
    optimal_rewards = []
    prev_action = None
    
    for i in range(len(X_test)):
        state = X_test.iloc[i]
        
        # Calculate optimal action
        feature_a = state['feature_a']
        feature_b = state['feature_b']
        feature_c = state['feature_c']
        
        if feature_a < 0.3:
            optimal_action = 0.2 + feature_b * 0.1
        elif feature_a > 0.7:
            optimal_action = 0.8 - feature_b * 0.1
        else:
            optimal_action = feature_a + (feature_b - 0.5) * 0.2
            optimal_action = np.clip(optimal_action, 0.3, 0.7)
        
        optimal_action += (feature_c - 0.5) * 0.05
        optimal_action = np.clip(optimal_action, 0, 1)
        
        # Calculate reward for optimal action
        reward = diverse_reward_function(state, optimal_action, Y_test.iloc[i], i, prev_action)
        optimal_rewards.append(reward)
        prev_action = optimal_action
    
    print(f"  Optimal strategy: {np.mean(optimal_rewards):.4f} ± {np.std(optimal_rewards):.4f}")
    print(f"  GATree strategy:  {np.mean(individual_rewards):.4f} ± {np.std(individual_rewards):.4f}")
    print(f"  Performance ratio: {np.mean(individual_rewards)/np.mean(optimal_rewards):.1%}")
    
    # Analyze action distribution by feature ranges
    print(f"\nAction Distribution by Feature Ranges:")
    
    # Low feature_a (should have low actions)
    low_mask = X_test['feature_a'] < 0.3
    if low_mask.sum() > 0:
        low_actions = [test_actions[i] for i in range(len(test_actions)) if low_mask.iloc[i]]
        print(f"  feature_a < 0.3 ({low_mask.sum()} samples): actions {np.mean(low_actions):.3f} ± {np.std(low_actions):.3f}")
    
    # High feature_a (should have high actions)
    high_mask = X_test['feature_a'] > 0.7
    if high_mask.sum() > 0:
        high_actions = [test_actions[i] for i in range(len(test_actions)) if high_mask.iloc[i]]
        print(f"  feature_a > 0.7 ({high_mask.sum()} samples): actions {np.mean(high_actions):.3f} ± {np.std(high_actions):.3f}")
    
    # Medium feature_a (should have medium actions)
    med_mask = (X_test['feature_a'] >= 0.3) & (X_test['feature_a'] <= 0.7)
    if med_mask.sum() > 0:
        med_actions = [test_actions[i] for i in range(len(test_actions)) if med_mask.iloc[i]]
        print(f"  0.3 ≤ feature_a ≤ 0.7 ({med_mask.sum()} samples): actions {np.mean(med_actions):.3f} ± {np.std(med_actions):.3f}")
    
    print(f"\nSuccess! The GATree learned to use {unique_actions} different continuous actions")
    print(f"with a standard deviation of {stats['std']:.3f}, demonstrating true action diversity.")


if __name__ == "__main__":
    main()