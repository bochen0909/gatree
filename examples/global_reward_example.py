#!/usr/bin/env python3
"""
Global Reward Function Example

Demonstrates how to use the global_reward_function parameter in GATreeActionSelector
to add global objectives like reward smoothness, action consistency, etc.
"""

import pandas as pd
import numpy as np
from gatree.methods.gatreeactionselector import GATreeActionSelector


def generate_test_data(n_timesteps=30, random_state=42):
    """Generate simple test data."""
    np.random.seed(random_state)
    
    X = pd.DataFrame({
        'demand': np.random.uniform(5, 15, n_timesteps),
        'cost': np.random.uniform(1, 2, n_timesteps)
    })
    
    Y = pd.DataFrame({
        'value': np.random.uniform(8, 12, n_timesteps)
    })
    
    return X, Y


def simple_reward(state, action, y_data, timestep):
    """Basic per-timestep reward function."""
    return float(y_data['value']) * action - float(state['cost']) * action


def smoothness_global_reward(rewards, actions, X, Y):
    """
    Global reward function that encourages smooth reward patterns.
    
    Args:
        rewards: List of individual rewards for each timestep
        actions: List of actions taken at each timestep  
        X: Feature data (pandas DataFrame)
        Y: Target data (pandas DataFrame)
    
    Returns:
        float: Global reward (positive = good, negative = penalty)
    """
    if len(rewards) < 2:
        return 0.0
    
    # Calculate reward volatility (lower is better)
    volatility = np.std(rewards)
    
    # Reward for low volatility (smooth rewards)
    smoothness_reward = max(0, 5 - volatility)
    
    # Penalty for big drops
    reward_diffs = np.diff(rewards)
    big_drops = np.sum(np.maximum(0, -reward_diffs - 2))  # Penalize drops > 2
    drop_penalty = big_drops * 2
    
    return smoothness_reward - drop_penalty


def consistency_global_reward(rewards, actions, X, Y):
    """
    Global reward function that encourages consistent actions.
    """
    if len(actions) < 2:
        return 0.0
    
    # Count action changes
    action_changes = sum(1 for i in range(1, len(actions)) if actions[i] != actions[i-1])
    
    # Reward for fewer changes
    consistency_reward = max(0, 10 - action_changes)
    
    return consistency_reward


def trend_global_reward(rewards, actions, X, Y):
    """
    Global reward function that encourages upward trending rewards.
    """
    if len(rewards) < 3:
        return 0.0
    
    # Calculate linear trend
    x = np.arange(len(rewards))
    trend_slope = np.polyfit(x, rewards, 1)[0]
    
    # Reward positive trends
    trend_reward = max(0, trend_slope * 10)
    
    return trend_reward


def demo_global_rewards():
    """Demonstrate global reward functionality."""
    print("Global Reward Function Demo")
    print("=" * 40)
    
    # Generate test data
    X, Y = generate_test_data(n_timesteps=25, random_state=42)
    
    print(f"Dataset: {len(X)} timesteps")
    print("Action space: [0, 1, 2, 3]")
    
    # Test configurations
    configs = [
        {
            'name': 'No Global Reward',
            'global_reward_function': None,
            'global_reward_weight': 0.0
        },
        {
            'name': 'Smoothness Reward',
            'global_reward_function': smoothness_global_reward,
            'global_reward_weight': 0.2
        },
        {
            'name': 'Consistency Reward',
            'global_reward_function': consistency_global_reward,
            'global_reward_weight': 0.1
        },
        {
            'name': 'Trend Reward',
            'global_reward_function': trend_global_reward,
            'global_reward_weight': 0.3
        }
    ]
    
    results = {}
    
    for config in configs:
        print(f"\nTraining: {config['name']}")
        print("-" * 25)
        
        selector = GATreeActionSelector(
            action_space=[0, 1, 2, 3],
            reward_function=simple_reward,
            global_reward_function=config['global_reward_function'],
            global_reward_weight=config['global_reward_weight'],
            max_depth=3,
            discount_factor=0.9,
            random_state=42
        )
        
        print(f"Selector: {selector}")
        
        # Train with small parameters for demo
        selector.fit(X, Y, population_size=10, max_iter=15)
        
        # Evaluate the strategy
        actions = []
        rewards = []
        
        for i in range(len(X)):
            action = selector.predict_action(X.iloc[i])
            reward = simple_reward(X.iloc[i], action, Y.iloc[i], i)
            actions.append(action)
            rewards.append(reward)
        
        # Calculate metrics
        total_reward = sum(rewards)
        reward_volatility = np.std(rewards) if len(rewards) > 1 else 0
        action_changes = sum(1 for i in range(1, len(actions)) if actions[i] != actions[i-1])
        
        # Calculate trend
        if len(rewards) >= 3:
            trend_slope = np.polyfit(range(len(rewards)), rewards, 1)[0]
        else:
            trend_slope = 0
        
        results[config['name']] = {
            'total_reward': total_reward,
            'reward_volatility': reward_volatility,
            'action_changes': action_changes,
            'trend_slope': trend_slope,
            'actions': actions,
            'rewards': rewards
        }
        
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Reward volatility: {reward_volatility:.3f}")
        print(f"  Action changes: {action_changes}")
        print(f"  Trend slope: {trend_slope:.3f}")
    
    # Compare results
    print(f"\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    
    print("Strategy\t\tTotal\tVolatility\tChanges\tTrend")
    print("-" * 60)
    
    for name, result in results.items():
        print(f"{name[:15]:15s}\t{result['total_reward']:.1f}\t{result['reward_volatility']:.3f}\t\t{result['action_changes']}\t{result['trend_slope']:.3f}")
    
    # Show action and reward sequences
    print(f"\nAction Sequences:")
    print("-" * 30)
    for name, result in results.items():
        actions_str = str(result['actions'][:12])
        print(f"{name[:15]:15s}: {actions_str}")
    
    print(f"\nReward Sequences (first 12):")
    print("-" * 35)
    for name, result in results.items():
        rewards_str = [f"{r:.1f}" for r in result['rewards'][:12]]
        print(f"{name[:15]:15s}: {', '.join(rewards_str)}")
    
    return results


if __name__ == "__main__":
    results = demo_global_rewards()
    
    print(f"\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print("Global reward functions allow you to:")
    print("✓ Encourage smooth reward patterns")
    print("✓ Promote action consistency")
    print("✓ Reward positive trends")
    print("✓ Add any global objective you can define")
    print(f"\nUsage: Just pass global_reward_function parameter!")
    print("No subclassing or fitness function rewriting needed.")