#!/usr/bin/env python3
"""
Discount Direction Comparison Demo

This example demonstrates the difference between forward and backward discounting
in GATreeActionSelector, showing how each approach prioritizes different parts
of the time series.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import the GATreeActionSelector
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from gatree.methods.gatreeactionselector import GATreeActionSelector


def generate_test_data(n_timesteps=100, random_state=42):
    """
    Generate synthetic data with different reward patterns over time.
    """
    np.random.seed(random_state)
    
    # Create features
    X = pd.DataFrame({
        'feature_1': np.random.randn(n_timesteps),
        'feature_2': np.random.randn(n_timesteps),
        'timestep': np.arange(n_timesteps)
    })
    
    # Create reward data with time-varying patterns
    Y = pd.DataFrame({
        'base_reward': 10 + 5 * np.sin(2 * np.pi * np.arange(n_timesteps) / 20),
        'early_bonus': np.maximum(0, 10 - 0.1 * np.arange(n_timesteps)),  # Decreases over time
        'late_bonus': np.maximum(0, -5 + 0.15 * np.arange(n_timesteps)),  # Increases over time
        'cost': 2 + np.random.uniform(0, 1, n_timesteps)
    })
    
    return X, Y


def time_sensitive_reward(state, action, y_data, timestep, previous_action):
    """
    Reward function that has different patterns over time.
    
    Action 0: Gets early bonus (good at beginning)
    Action 1: Gets late bonus (good at end)  
    Action 2: Consistent but lower reward
    """
    base_reward = y_data['base_reward']
    early_bonus = y_data['early_bonus']
    late_bonus = y_data['late_bonus']
    cost = y_data['cost']
    
    # Base reward calculation
    if action == 0:
        # Early strategy: high reward early, low reward late
        reward = base_reward + early_bonus - cost
    elif action == 1:
        # Late strategy: low reward early, high reward late
        reward = base_reward + late_bonus - cost
    else:  # action == 2
        # Consistent strategy: moderate reward throughout
        reward = base_reward * 0.7 - cost * 0.8
    
    # Optional: Add small penalty for frequent strategy changes
    if previous_action is not None and previous_action != action:
        reward -= 0.5  # Penalty for changing strategies
    
    return reward
        return base_reward + early_bonus - cost
    elif action == 1:
        # Late strategy: low reward early, high reward late
        return base_reward + late_bonus - cost
    else:  # action == 2
        # Consistent strategy: moderate reward throughout
        return base_reward * 0.8 - cost * 0.5


def analyze_discount_impact(discount_factor=0.9, n_timesteps=100):
    """
    Analyze the impact of forward vs backward discounting.
    """
    print(f"Discount Impact Analysis (factor={discount_factor}, timesteps={n_timesteps})")
    print("=" * 70)
    
    # Calculate discount weights
    forward_weights = [discount_factor ** t for t in range(n_timesteps)]
    backward_weights = [discount_factor ** (n_timesteps - 1 - t) for t in range(n_timesteps)]
    
    # Show weights at key timesteps
    print("Discount weights at key timesteps:")
    print("Timestep\tForward\t\tBackward\tRatio (B/F)")
    print("-" * 50)
    for t in [0, 10, 25, 50, 75, 90, n_timesteps-1]:
        if t < n_timesteps:
            forward_w = forward_weights[t]
            backward_w = backward_weights[t]
            ratio = backward_w / forward_w if forward_w > 0 else float('inf')
            print(f"{t:3d}\t\t{forward_w:.4f}\t\t{backward_w:.4f}\t\t{ratio:.1f}x")
    
    # Calculate cumulative weights
    forward_total = sum(forward_weights)
    backward_total = sum(backward_weights)
    
    # Analyze early vs late emphasis
    mid_point = n_timesteps // 2
    forward_early = sum(forward_weights[:mid_point])
    forward_late = sum(forward_weights[mid_point:])
    backward_early = sum(backward_weights[:mid_point])
    backward_late = sum(backward_weights[mid_point:])
    
    print(f"\nCumulative weight analysis:")
    print(f"Forward  - Early half: {forward_early:.2f} ({forward_early/forward_total:.1%}), Late half: {forward_late:.2f} ({forward_late/forward_total:.1%})")
    print(f"Backward - Early half: {backward_early:.2f} ({backward_early/backward_total:.1%}), Late half: {backward_late:.2f} ({backward_late/backward_total:.1%})")
    
    return forward_weights, backward_weights


def compare_strategies():
    """
    Compare forward vs backward discounting strategies.
    """
    print("GATree Discount Direction Comparison")
    print("=" * 50)
    
    # Generate test data
    X, Y = generate_test_data(n_timesteps=80, random_state=42)
    
    print(f"Dataset: {len(X)} timesteps, {X.shape[1]} features")
    print(f"Action space: [0, 1, 2]")
    print("- Action 0: Early bonus strategy")
    print("- Action 1: Late bonus strategy") 
    print("- Action 2: Consistent strategy")
    
    # Analyze discount impact
    print(f"\n")
    forward_weights, backward_weights = analyze_discount_impact(discount_factor=0.9, n_timesteps=len(X))
    
    # Train both forward and backward models
    action_space = [0, 1, 2]
    
    print(f"\nTraining Forward Discounting Model...")
    forward_selector = GATreeActionSelector(
        action_space=action_space,
        reward_function=time_sensitive_reward,
        max_depth=4,
        discount_factor=0.9,
        discount_direction='forward',
        random_state=42
    )
    
    forward_selector.fit(
        X=X, Y=Y,
        population_size=20,
        max_iter=30,
        mutation_probability=0.2
    )
    
    print(f"Training Backward Discounting Model...")
    backward_selector = GATreeActionSelector(
        action_space=action_space,
        reward_function=time_sensitive_reward,
        max_depth=4,
        discount_factor=0.9,
        discount_direction='backward',
        random_state=42
    )
    
    backward_selector.fit(
        X=X, Y=Y,
        population_size=20,
        max_iter=30,
        mutation_probability=0.2
    )
    
    # Evaluate both models
    print(f"\nEvaluating Models...")
    
    # Get action sequences
    forward_actions = []
    backward_actions = []
    
    for i in range(len(X)):
        forward_action = forward_selector.predict_action(X.iloc[i])
        backward_action = backward_selector.predict_action(X.iloc[i])
        forward_actions.append(forward_action)
        backward_actions.append(backward_action)
    
    # Calculate rewards for both strategies
    forward_rewards = []
    backward_rewards = []
    
    for i in range(len(X)):
        forward_reward = time_sensitive_reward(X.iloc[i], forward_actions[i], Y.iloc[i], i)
        backward_reward = time_sensitive_reward(X.iloc[i], backward_actions[i], Y.iloc[i], i)
        forward_rewards.append(forward_reward)
        backward_rewards.append(backward_reward)
    
    # Calculate discounted totals
    forward_discounted_total = sum(r * w for r, w in zip(forward_rewards, forward_weights))
    backward_discounted_total = sum(r * w for r, w in zip(backward_rewards, backward_weights))
    
    # Calculate undiscounted totals
    forward_undiscounted_total = sum(forward_rewards)
    backward_undiscounted_total = sum(backward_rewards)
    
    print(f"\nResults Comparison:")
    print("=" * 50)
    print(f"Forward Discounting:")
    print(f"  Discounted total reward: {forward_discounted_total:.2f}")
    print(f"  Undiscounted total reward: {forward_undiscounted_total:.2f}")
    print(f"  Average reward per timestep: {forward_undiscounted_total/len(X):.3f}")
    
    print(f"Backward Discounting:")
    print(f"  Discounted total reward: {backward_discounted_total:.2f}")
    print(f"  Undiscounted total reward: {backward_undiscounted_total:.2f}")
    print(f"  Average reward per timestep: {backward_undiscounted_total/len(X):.3f}")
    
    # Action distribution analysis
    forward_dist = {action: forward_actions.count(action) for action in action_space}
    backward_dist = {action: backward_actions.count(action) for action in action_space}
    
    print(f"\nAction Distribution:")
    print("Action\tForward\tBackward\tDifference")
    print("-" * 40)
    for action in action_space:
        forward_count = forward_dist[action]
        backward_count = backward_dist[action]
        forward_pct = (forward_count / len(forward_actions)) * 100
        backward_pct = (backward_count / len(backward_actions)) * 100
        diff = backward_pct - forward_pct
        print(f"{action}\t{forward_pct:.1f}%\t{backward_pct:.1f}%\t\t{diff:+.1f}%")
    
    # Temporal analysis - divide into early, middle, late periods
    n_periods = 3
    period_size = len(X) // n_periods
    
    print(f"\nTemporal Action Analysis:")
    print("Period\t\tForward\t\tBackward")
    print("-" * 45)
    
    for period in range(n_periods):
        start_idx = period * period_size
        end_idx = start_idx + period_size if period < n_periods - 1 else len(X)
        
        period_name = ["Early", "Middle", "Late"][period]
        
        forward_period_actions = forward_actions[start_idx:end_idx]
        backward_period_actions = backward_actions[start_idx:end_idx]
        
        forward_period_dist = {action: forward_period_actions.count(action) for action in action_space}
        backward_period_dist = {action: backward_period_actions.count(action) for action in action_space}
        
        # Find most common action in each period
        forward_most_common = max(forward_period_dist, key=forward_period_dist.get)
        backward_most_common = max(backward_period_dist, key=backward_period_dist.get)
        
        forward_pct = (forward_period_dist[forward_most_common] / len(forward_period_actions)) * 100
        backward_pct = (backward_period_dist[backward_most_common] / len(backward_period_actions)) * 100
        
        print(f"{period_name:8s}\tAction {forward_most_common} ({forward_pct:.0f}%)\tAction {backward_most_common} ({backward_pct:.0f}%)")
    
    # Show sample decisions from different time periods
    print(f"\nSample Decisions:")
    print("Timestep\tForward\tBackward\tForward Reward\tBackward Reward")
    print("-" * 65)
    
    sample_timesteps = [0, 10, 20, 40, 60, len(X)-1]
    for t in sample_timesteps:
        if t < len(X):
            f_action = forward_actions[t]
            b_action = backward_actions[t]
            f_reward = forward_rewards[t]
            b_reward = backward_rewards[t]
            print(f"{t:3d}\t\t{f_action}\t{b_action}\t{f_reward:.2f}\t\t{b_reward:.2f}")
    
    # Model information
    print(f"\nModel Information:")
    print(f"Forward Model:  {forward_selector}")
    print(f"Backward Model: {backward_selector}")
    
    # Training evolution comparison
    print(f"\nTraining Evolution:")
    print(f"Forward  - Initial: {forward_selector._best_rewards[0]:.2f}, Final: {forward_selector._best_rewards[-1]:.2f}")
    print(f"Backward - Initial: {backward_selector._best_rewards[0]:.2f}, Final: {backward_selector._best_rewards[-1]:.2f}")
    
    # Plot results if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Discount weights comparison
        timesteps = range(len(X))
        ax1.plot(timesteps, forward_weights, label='Forward', color='blue', alpha=0.7)
        ax1.plot(timesteps, backward_weights, label='Backward', color='red', alpha=0.7)
        ax1.set_xlabel('Timestep')
        ax1.set_ylabel('Discount Weight')
        ax1.set_title('Discount Weights Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Action sequences
        ax2.plot(timesteps, forward_actions, label='Forward', color='blue', alpha=0.7, marker='o', markersize=3)
        ax2.plot(timesteps, backward_actions, label='Backward', color='red', alpha=0.7, marker='s', markersize=3)
        ax2.set_xlabel('Timestep')
        ax2.set_ylabel('Action')
        ax2.set_title('Action Sequences')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yticks(action_space)
        
        # Cumulative rewards
        forward_cumulative = np.cumsum(forward_rewards)
        backward_cumulative = np.cumsum(backward_rewards)
        
        ax3.plot(timesteps, forward_cumulative, label='Forward', color='blue', alpha=0.8)
        ax3.plot(timesteps, backward_cumulative, label='Backward', color='red', alpha=0.8)
        ax3.set_xlabel('Timestep')
        ax3.set_ylabel('Cumulative Reward')
        ax3.set_title('Cumulative Rewards (Undiscounted)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Training evolution
        ax4.plot(forward_selector._best_rewards, label='Forward', color='blue', alpha=0.8)
        ax4.plot(backward_selector._best_rewards, label='Backward', color='red', alpha=0.8)
        ax4.set_xlabel('Generation')
        ax4.set_ylabel('Best Reward')
        ax4.set_title('Training Evolution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("\nMatplotlib not available - skipping plots")
    
    return forward_selector, backward_selector


def demonstrate_use_cases():
    """
    Demonstrate when to use forward vs backward discounting.
    """
    print(f"\n" + "=" * 60)
    print("WHEN TO USE EACH DISCOUNT DIRECTION")
    print("=" * 60)
    
    print("FORWARD DISCOUNTING (Standard):")
    print("✓ Financial planning - early returns are more valuable")
    print("✓ Real-time systems - immediate performance matters most")
    print("✓ Resource allocation - early efficiency gains compound")
    print("✓ Risk management - future uncertainty increases over time")
    print("✓ Online learning - adapt quickly to current conditions")
    
    print(f"\nBACKWARD DISCOUNTING (Reverse):")
    print("✓ Final exam preparation - end performance matters most")
    print("✓ System warm-up - later performance more representative")
    print("✓ Long-term optimization - steady-state behavior important")
    print("✓ Quality improvement - final output quality critical")
    print("✓ Convergence problems - solution quality at end matters")
    
    print(f"\nEXAMPLE SCENARIOS:")
    print("Forward:  Trading algorithm (early profits compound)")
    print("Backward: Manufacturing process (final quality matters)")
    print("Forward:  Emergency response (immediate action critical)")
    print("Backward: Research project (final results matter most)")


if __name__ == "__main__":
    # Run the comparison
    forward_model, backward_model = compare_strategies()
    
    # Show use cases
    demonstrate_use_cases()
    
    print(f"\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("Key takeaway: Discount direction fundamentally changes")
    print("which part of the time series the model optimizes for!")