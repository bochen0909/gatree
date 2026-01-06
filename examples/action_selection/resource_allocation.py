"""
Example of using GATreeActionSelector for resource allocation optimization.

This example demonstrates how to use the evolutionary decision tree action selector
to optimize resource allocation decisions over time based on demand patterns and costs.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import the GATreeActionSelector
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from gatree.methods.gatreeactionselector import GATreeActionSelector


def generate_resource_demand_data(n_timesteps=250, random_state=42):
    """
    Generate synthetic resource demand and system state data.
    
    Returns:
        tuple: (features_df, cost_data_df) for resource allocation decisions
    """
    np.random.seed(random_state)
    
    # Generate time-based patterns
    time = np.arange(n_timesteps)
    
    # Base demand with daily and weekly patterns
    daily_pattern = 10 + 5 * np.sin(2 * np.pi * time / 24)  # Daily cycle
    weekly_pattern = 2 * np.sin(2 * np.pi * time / (24 * 7))  # Weekly cycle
    trend = 0.01 * time  # Slight upward trend
    noise = np.random.normal(0, 2, n_timesteps)
    
    base_demand = daily_pattern + weekly_pattern + trend + noise
    base_demand = np.maximum(base_demand, 1)  # Ensure positive demand
    
    # System features
    features = pd.DataFrame()
    
    # Demand-related features
    features['current_demand'] = base_demand
    features['demand_ma_5'] = pd.Series(base_demand).rolling(window=5, min_periods=1).mean()
    features['demand_ma_24'] = pd.Series(base_demand).rolling(window=24, min_periods=1).mean()
    features['demand_trend'] = pd.Series(base_demand).diff().fillna(0)
    features['demand_volatility'] = pd.Series(base_demand).rolling(window=12, min_periods=1).std().fillna(0)
    
    # Time-based features
    features['hour_of_day'] = time % 24
    features['day_of_week'] = (time // 24) % 7
    features['is_weekend'] = (features['day_of_week'] >= 5).astype(int)
    features['is_peak_hour'] = ((features['hour_of_day'] >= 8) & (features['hour_of_day'] <= 18)).astype(int)
    
    # System capacity and utilization
    max_capacity = 20
    features['max_capacity'] = max_capacity
    features['capacity_utilization'] = np.minimum(base_demand / max_capacity, 1.0)
    
    # Historical allocation (for context)
    features['prev_allocation'] = np.concatenate([[5], np.random.randint(0, 6, n_timesteps-1)])
    features['allocation_efficiency'] = np.minimum(features['prev_allocation'] / features['current_demand'], 2.0)
    
    # Resource availability (varies over time)
    availability_factor = 0.8 + 0.2 * np.sin(2 * np.pi * time / 100) + 0.1 * np.random.random(n_timesteps)
    features['resource_availability'] = availability_factor
    
    # Cost data for reward calculation
    cost_data = pd.DataFrame()
    
    # Base cost per unit (varies with time and availability)
    base_cost = 1.0 + 0.3 * np.sin(2 * np.pi * time / 50) + 0.1 * np.random.random(n_timesteps)
    cost_data['unit_cost'] = base_cost / availability_factor  # Higher cost when less available
    
    # Demand satisfaction value
    cost_data['satisfaction_value'] = 10 + 2 * np.random.random(n_timesteps)
    
    # Penalty for over/under allocation
    cost_data['overallocation_penalty'] = 2.0
    cost_data['underallocation_penalty'] = 5.0
    
    # Demand for reward calculation
    cost_data['actual_demand'] = base_demand
    
    return features, cost_data


def resource_allocation_reward(state, action, cost_data, timestep):
    """
    Calculate reward for a resource allocation action.
    
    Args:
        state: Current system state (feature vector)
        action: Resource allocation level (0-5)
        cost_data: Cost and demand information
        timestep: Current timestep
    
    Returns:
        float: Reward for the allocation decision
    """
    allocation = action
    demand = cost_data['actual_demand']
    unit_cost = cost_data['unit_cost']
    satisfaction_value = cost_data['satisfaction_value']
    overallocation_penalty = cost_data['overallocation_penalty']
    underallocation_penalty = cost_data['underallocation_penalty']
    
    # Base cost of allocation
    allocation_cost = allocation * unit_cost
    
    # Satisfaction reward (how well we meet demand)
    if allocation >= demand:
        # Full satisfaction
        satisfaction_reward = satisfaction_value
        # Penalty for over-allocation (waste)
        waste_penalty = (allocation - demand) * overallocation_penalty
    else:
        # Partial satisfaction
        satisfaction_ratio = allocation / demand if demand > 0 else 0
        satisfaction_reward = satisfaction_value * satisfaction_ratio
        # Penalty for under-allocation (unmet demand)
        shortage_penalty = (demand - allocation) * underallocation_penalty
        waste_penalty = shortage_penalty
    
    # Total reward
    reward = satisfaction_reward - allocation_cost - waste_penalty
    
    return reward


def main():
    """
    Main function to demonstrate GATreeActionSelector for resource allocation.
    """
    print("GATree Action Selector Example - Resource Allocation")
    print("=" * 58)
    
    # Generate synthetic resource demand data
    print("Generating synthetic resource demand data...")
    features, cost_data = generate_resource_demand_data(n_timesteps=400, random_state=42)
    
    print(f"Dataset shape: {features.shape}")
    print(f"Demand range: [{features['current_demand'].min():.2f}, {features['current_demand'].max():.2f}]")
    print(f"Average demand: {features['current_demand'].mean():.2f}")
    print(f"Cost range: [{cost_data['unit_cost'].min():.2f}, {cost_data['unit_cost'].max():.2f}]")
    
    # Split data into train/test
    train_size = int(0.75 * len(features))
    
    X_train = features.iloc[:train_size].copy()
    Y_train = cost_data.iloc[:train_size].copy()
    X_test = features.iloc[train_size:].copy()
    Y_test = cost_data.iloc[train_size:].copy()
    
    print(f"Training period: {train_size} timesteps")
    print(f"Testing period: {len(X_test)} timesteps")
    
    # Scale features for better performance
    scaler = StandardScaler()
    feature_cols = ['current_demand', 'demand_ma_5', 'demand_ma_24', 'demand_trend', 
                   'demand_volatility', 'capacity_utilization', 'allocation_efficiency',
                   'resource_availability']
    
    X_train_scaled = X_train.copy()
    X_train_scaled[feature_cols] = scaler.fit_transform(X_train[feature_cols])
    
    X_test_scaled = X_test.copy()
    X_test_scaled[feature_cols] = scaler.transform(X_test[feature_cols])
    
    # Define action space (resource allocation levels)
    action_space = [0, 1, 2, 3, 4, 5]  # 0 to 5 units of resources
    
    # Create and train the action selector
    print(f"\nTraining GATree Action Selector...")
    print(f"Action space: {action_space} (resource units)")
    print("Parameters: population_size=40, max_iter=60, max_depth=7")
    
    selector = GATreeActionSelector(
        action_space=action_space,
        reward_function=resource_allocation_reward,
        max_depth=7,
        discount_factor=0.95,  # Moderate discount for future rewards
        n_jobs=2,
        random_state=42
    )
    
    # Train the selector
    selector.fit(
        X=X_train_scaled,
        Y=Y_train,
        population_size=40,
        max_iter=60,
        mutation_probability=0.18,
        elite_size=3,
        selection_tournament_size=4
    )
    
    # Test the allocation strategy
    print("\nTesting resource allocation strategy...")
    
    # Simulate allocation decisions on test data
    total_reward, individual_rewards, actions = selector.simulate_rewards(
        X_test_scaled, Y_test, resource_allocation_reward
    )
    
    # Calculate performance metrics
    action_distribution = selector.get_action_distribution(X_test_scaled)
    
    print("\nAllocation Results:")
    print("=" * 35)
    print(f"Total reward: {total_reward:.2f}")
    print(f"Average reward per timestep: {total_reward / len(X_test):.4f}")
    print(f"Number of timesteps: {len(X_test)}")
    
    print(f"\nAllocation Distribution:")
    for action, count in action_distribution.items():
        percentage = (count / len(actions)) * 100
        print(f"  {action} units: {count} times ({percentage:.1f}%)")
    
    # Calculate allocation efficiency metrics
    test_demands = Y_test['actual_demand'].values
    allocations = actions
    
    # Satisfaction metrics
    total_demand = sum(test_demands)
    total_allocation = sum(allocations)
    
    satisfied_demand = sum(min(alloc, demand) for alloc, demand in zip(allocations, test_demands))
    satisfaction_rate = satisfied_demand / total_demand if total_demand > 0 else 0
    
    over_allocation = sum(max(0, alloc - demand) for alloc, demand in zip(allocations, test_demands))
    under_allocation = sum(max(0, demand - alloc) for alloc, demand in zip(allocations, test_demands))
    
    print(f"\nEfficiency Metrics:")
    print("-" * 25)
    print(f"Total demand: {total_demand:.2f}")
    print(f"Total allocation: {total_allocation:.2f}")
    print(f"Satisfaction rate: {satisfaction_rate:.1%}")
    print(f"Over-allocation: {over_allocation:.2f}")
    print(f"Under-allocation: {under_allocation:.2f}")
    print(f"Allocation efficiency: {satisfied_demand / total_allocation:.1%}" if total_allocation > 0 else "N/A")
    
    # Show sample allocation decisions
    print(f"\nSample Allocation Decisions:")
    print("-" * 60)
    print("Step\tDemand\tAllocation\tReward\tCost\tSatisfaction")
    for i in range(min(10, len(actions))):
        demand = test_demands[i]
        allocation = actions[i]
        reward = individual_rewards[i]
        cost = allocation * Y_test.iloc[i]['unit_cost']
        satisfaction = min(allocation, demand) / demand if demand > 0 else 1.0
        print(f"{i+1}\t{demand:.1f}\t{allocation}\t\t{reward:.2f}\t{cost:.2f}\t{satisfaction:.1%}")
    
    # Compare with simple strategies
    print(f"\nStrategy Comparison:")
    print("-" * 30)
    
    # Always allocate average demand
    avg_demand = features['current_demand'].mean()
    avg_allocation = int(round(avg_demand))
    avg_rewards = []
    for i in range(len(X_test)):
        reward = resource_allocation_reward(
            X_test.iloc[i], avg_allocation, Y_test.iloc[i], i
        )
        avg_rewards.append(reward)
    avg_total = sum(avg_rewards)
    
    # Always allocate maximum
    max_rewards = []
    for i in range(len(X_test)):
        reward = resource_allocation_reward(
            X_test.iloc[i], max(action_space), Y_test.iloc[i], i
        )
        max_rewards.append(reward)
    max_total = sum(max_rewards)
    
    # Always allocate minimum
    min_rewards = []
    for i in range(len(X_test)):
        reward = resource_allocation_reward(
            X_test.iloc[i], min(action_space), Y_test.iloc[i], i
        )
        min_rewards.append(reward)
    min_total = sum(min_rewards)
    
    print(f"GATree Strategy: {total_reward:.2f}")
    print(f"Always Average ({avg_allocation}): {avg_total:.2f}")
    print(f"Always Maximum ({max(action_space)}): {max_total:.2f}")
    print(f"Always Minimum ({min(action_space)}): {min_total:.2f}")
    
    best_baseline = max(avg_total, max_total, min_total)
    improvement = ((total_reward - best_baseline) / abs(best_baseline)) * 100 if best_baseline != 0 else 0
    print(f"Improvement vs best baseline: {improvement:.1f}%")
    
    # Show training evolution
    print(f"\nTraining Evolution:")
    print(f"Initial best reward: {selector._best_rewards[0]:.2f}")
    print(f"Final best reward: {selector._best_rewards[-1]:.2f}")
    print(f"Improvement: {selector._best_rewards[-1] - selector._best_rewards[0]:.2f}")
    
    # Tree information
    print(f"\nFinal Tree Information:")
    print(f"Tree depth: {selector._tree.max_depth()}")
    print(f"Tree size (nodes): {selector._tree.size()}")
    print(f"Number of leaves: {len(selector._tree.get_leaves())}")
    
    # Plot results if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Demand vs Allocation
        test_indices = range(len(X_test))
        ax1.plot(test_indices, test_demands, label='Demand', color='blue', alpha=0.7)
        ax1.plot(test_indices, allocations, label='Allocation', color='red', alpha=0.7)
        ax1.fill_between(test_indices, test_demands, allocations, alpha=0.3, color='gray')
        ax1.set_xlabel('Timestep')
        ax1.set_ylabel('Resource Units')
        ax1.set_title('Demand vs Allocation')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Cumulative rewards
        cumulative_rewards = np.cumsum(individual_rewards)
        ax2.plot(test_indices, cumulative_rewards, color='green')
        ax2.set_xlabel('Timestep')
        ax2.set_ylabel('Cumulative Reward')
        ax2.set_title('Cumulative Reward Over Time')
        ax2.grid(True, alpha=0.3)
        
        # Training evolution
        ax3.plot(selector._best_rewards, label='Best Reward', color='purple')
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Total Reward')
        ax3.set_title('Training Evolution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Allocation distribution
        allocations_list = list(action_distribution.keys())
        counts = list(action_distribution.values())
        ax4.bar(allocations_list, counts, color='orange', alpha=0.7)
        ax4.set_xlabel('Allocation Level')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Allocation Distribution')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("\nMatplotlib not available - skipping plots")
    
    print(f"\n{selector}")


if __name__ == "__main__":
    main()