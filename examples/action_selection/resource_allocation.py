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


def resource_allocation_reward(state, action, cost_data, timestep, allocation_multiplier=5):
    """
    Calculate reward for a resource allocation action.
    
    Args:
        state: Current system state (feature vector)
        action: Resource allocation level (0-5, multiplied by allocation_multiplier)
        cost_data: Cost and demand information
        timestep: Current timestep
        allocation_multiplier: Multiplier for actual resource units
    
    Returns:
        float: Reward for the allocation decision
    """
    allocation = action * allocation_multiplier  # Convert action to actual allocation
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
    
    # Define action space (resource allocation levels with multiplier)
    action_space = [0, 1, 2, 3, 4, 5]  # Base allocation levels
    allocation_multiplier = 5  # Each action represents 5 units
    
    print(f"Action space: {action_space} (each unit = {allocation_multiplier} resources)")
    print(f"Effective allocation range: 0 to {max(action_space) * allocation_multiplier} units")
    
    # Create and train the action selector
    print(f"\nTraining GATree Action Selector...")
    print("Parameters: population_size=40, max_iter=60, max_depth=7")
    
    # Create a wrapper function that includes the multiplier
    def reward_function_with_multiplier(state, action, cost_data, timestep):
        return resource_allocation_reward(state, action, cost_data, timestep, allocation_multiplier)
    
    # Optional: Define a global reward function for smoother allocation
    def allocation_smoothness_reward(rewards, actions, X, Y):
        """Global reward that encourages smoother reward patterns."""
        if len(rewards) < 2:
            return 0.0
        
        # Reward for lower volatility
        volatility = np.std(rewards)
        smoothness_bonus = max(0, 10 - volatility)
        
        # Small penalty for frequent allocation changes
        if len(actions) > 1:
            changes = sum(1 for i in range(1, len(actions)) if actions[i] != actions[i-1])
            change_penalty = changes * 0.5
        else:
            change_penalty = 0
        
        return smoothness_bonus - change_penalty
    
    selector = GATreeActionSelector(
        action_space=action_space,
        reward_function=reward_function_with_multiplier,
        max_depth=7,
        discount_factor=0.95,  # Moderate discount for future rewards
        discount_direction='forward',  # Can be changed to 'backward' for end-focused optimization
        global_reward_function=None,  # Set to allocation_smoothness_reward for smoother allocation
        global_reward_weight=0.1,  # Weight for global reward component
        n_jobs=2,
        random_state=42
    )
    
    print(f"Discount direction: {selector.discount_direction}")
    if selector.discount_direction == 'backward':
        print("→ Optimizing for end-of-period performance (steady-state)")
    else:
        print("→ Optimizing for early performance (immediate efficiency)")
    
    if selector.global_reward_function is not None:
        print(f"→ Using global reward function with weight {selector.global_reward_weight}")
    else:
        print("→ No global reward function (standard optimization)")
    
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
    
    # Evaluate the allocation strategy on both train and test data
    print("\nEvaluating resource allocation strategy...")
    
    # Evaluate on training data
    print("Evaluating on training data...")
    train_total_reward, train_individual_rewards, train_actions = selector.simulate_rewards(
        X_train_scaled, Y_train, reward_function_with_multiplier
    )
    train_action_distribution = selector.get_action_distribution(X_train_scaled)
    
    # Evaluate on test data
    print("Evaluating on test data...")
    test_total_reward, test_individual_rewards, test_actions = selector.simulate_rewards(
        X_test_scaled, Y_test, reward_function_with_multiplier
    )
    test_action_distribution = selector.get_action_distribution(X_test_scaled)
    
    print("\nAllocation Results:")
    print("=" * 50)
    
    # Training results
    print("TRAINING SET:")
    print(f"  Total reward: {train_total_reward:.2f}")
    print(f"  Average reward per timestep: {train_total_reward / len(X_train):.4f}")
    print(f"  Number of timesteps: {len(X_train)}")
    
    # Test results
    print("TEST SET:")
    print(f"  Total reward: {test_total_reward:.2f}")
    print(f"  Average reward per timestep: {test_total_reward / len(X_test):.4f}")
    print(f"  Number of timesteps: {len(X_test)}")
    
    # Performance comparison
    train_avg_reward = train_total_reward / len(X_train)
    test_avg_reward = test_total_reward / len(X_test)
    generalization_gap = train_avg_reward - test_avg_reward
    generalization_ratio = test_avg_reward / train_avg_reward if train_avg_reward != 0 else 0
    
    print(f"\nGENERALIZATION ANALYSIS:")
    print(f"  Generalization gap: {generalization_gap:.4f}")
    print(f"  Test/Train ratio: {generalization_ratio:.3f}")
    if generalization_ratio > 0.9:
        print("  → Excellent generalization")
    elif generalization_ratio > 0.8:
        print("  → Good generalization")
    elif generalization_ratio > 0.7:
        print("  → Moderate generalization")
    else:
        print("  → Poor generalization (possible overfitting)")

    print(f"\nAllocation Distribution Comparison:")
    print("Action\tTrain\tTest\tDifference\tActual Units")
    print("-" * 50)
    for action in sorted(set(list(train_action_distribution.keys()) + list(test_action_distribution.keys()))):
        train_count = train_action_distribution.get(action, 0)
        test_count = test_action_distribution.get(action, 0)
        train_pct = (train_count / len(train_actions)) * 100
        test_pct = (test_count / len(test_actions)) * 100
        diff = test_pct - train_pct
        actual_units = action * allocation_multiplier
        print(f"{action}\t{train_pct:.1f}%\t{test_pct:.1f}%\t{diff:+.1f}%\t\t{actual_units}")
    
    # Calculate allocation efficiency metrics for both datasets
    def calculate_efficiency_metrics(demands, actions, dataset_name):
        # Convert actions to actual allocations
        allocations = [action * allocation_multiplier for action in actions]
        
        total_demand = sum(demands)
        total_allocation = sum(allocations)
        
        satisfied_demand = sum(min(alloc, demand) for alloc, demand in zip(allocations, demands))
        satisfaction_rate = satisfied_demand / total_demand if total_demand > 0 else 0
        
        over_allocation = sum(max(0, alloc - demand) for alloc, demand in zip(allocations, demands))
        under_allocation = sum(max(0, demand - alloc) for alloc, demand in zip(allocations, demands))
        
        print(f"\n{dataset_name.upper()} Efficiency Metrics:")
        print("-" * 30)
        print(f"Total demand: {total_demand:.2f}")
        print(f"Total allocation: {total_allocation:.2f}")
        print(f"Satisfaction rate: {satisfaction_rate:.1%}")
        print(f"Over-allocation: {over_allocation:.2f}")
        print(f"Under-allocation: {under_allocation:.2f}")
        print(f"Allocation efficiency: {satisfied_demand / total_allocation:.1%}" if total_allocation > 0 else "N/A")
        
        return {
            'total_demand': total_demand,
            'total_allocation': total_allocation,
            'satisfaction_rate': satisfaction_rate,
            'over_allocation': over_allocation,
            'under_allocation': under_allocation,
            'allocation_efficiency': satisfied_demand / total_allocation if total_allocation > 0 else 0
        }
    
    # Calculate metrics for both datasets
    train_demands = Y_train['actual_demand'].values
    test_demands = Y_test['actual_demand'].values
    
    train_metrics = calculate_efficiency_metrics(train_demands, train_actions, "Training")
    test_metrics = calculate_efficiency_metrics(test_demands, test_actions, "Test")
    
    # Show sample allocation decisions from test set
    print(f"\nSample Test Set Allocation Decisions:")
    print("-" * 70)
    print("Step\tDemand\tAction\tActual\tReward\tCost\tSatisfaction")
    print("\t\t\tAlloc")
    for i in range(min(10, len(test_actions))):
        demand = test_demands[i]
        action = test_actions[i]
        actual_allocation = action * allocation_multiplier
        reward = test_individual_rewards[i]
        cost = actual_allocation * Y_test.iloc[i]['unit_cost']
        satisfaction = min(actual_allocation, demand) / demand if demand > 0 else 1.0
        print(f"{i+1}\t{demand:.1f}\t{action}\t{actual_allocation}\t{reward:.2f}\t{cost:.2f}\t{satisfaction:.1%}")
    
    # Compare with simple strategies on both datasets
    print(f"\nStrategy Comparison:")
    print("-" * 50)
    
    def evaluate_baseline_strategy(X_data, Y_data, allocation_action, strategy_name):
        rewards = []
        for i in range(len(X_data)):
            reward = reward_function_with_multiplier(
                X_data.iloc[i], allocation_action, Y_data.iloc[i], i
            )
            rewards.append(reward)
        return sum(rewards)
    
    # Calculate baselines for both datasets
    avg_demand = features['current_demand'].mean()
    avg_allocation_action = int(round(avg_demand / allocation_multiplier))  # Convert to action space
    avg_allocation_action = max(0, min(avg_allocation_action, max(action_space)))  # Clamp to valid range
    
    # Training baselines
    train_avg_total = evaluate_baseline_strategy(X_train, Y_train, avg_allocation_action, "Average")
    train_max_total = evaluate_baseline_strategy(X_train, Y_train, max(action_space), "Maximum")
    train_min_total = evaluate_baseline_strategy(X_train, Y_train, min(action_space), "Minimum")
    
    # Test baselines
    test_avg_total = evaluate_baseline_strategy(X_test, Y_test, avg_allocation_action, "Average")
    test_max_total = evaluate_baseline_strategy(X_test, Y_test, max(action_space), "Maximum")
    test_min_total = evaluate_baseline_strategy(X_test, Y_test, min(action_space), "Minimum")
    
    print("Strategy\t\tTrain\t\tTest\t\tGeneralization")
    print("-" * 65)
    print(f"GATree\t\t\t{train_total_reward:.2f}\t\t{test_total_reward:.2f}\t\t{test_total_reward/train_total_reward:.3f}")
    avg_actual = avg_allocation_action * allocation_multiplier
    max_actual = max(action_space) * allocation_multiplier
    min_actual = min(action_space) * allocation_multiplier
    print(f"Always Average ({avg_actual})\t{train_avg_total:.2f}\t\t{test_avg_total:.2f}\t\t{test_avg_total/train_avg_total:.3f}")
    print(f"Always Maximum ({max_actual})\t{train_max_total:.2f}\t\t{test_max_total:.2f}\t\t{test_max_total/train_max_total:.3f}")
    print(f"Always Minimum ({min_actual})\t{train_min_total:.2f}\t\t{test_min_total:.2f}\t\t{test_min_total/train_min_total:.3f}")
    
    # Best baseline comparison
    best_train_baseline = max(train_avg_total, train_max_total, train_min_total)
    best_test_baseline = max(test_avg_total, test_max_total, test_min_total)
    
    train_improvement = ((train_total_reward - best_train_baseline) / abs(best_train_baseline)) * 100 if best_train_baseline != 0 else 0
    test_improvement = ((test_total_reward - best_test_baseline) / abs(best_test_baseline)) * 100 if best_test_baseline != 0 else 0
    
    print(f"\nImprovement vs best baseline:")
    print(f"  Training: {train_improvement:.1f}%")
    print(f"  Test: {test_improvement:.1f}%")
    
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
        
        # Demand vs Allocation (Test Set)
        test_indices = range(len(X_test))
        test_actual_allocations = [action * allocation_multiplier for action in test_actions]
        
        ax1.plot(test_indices, test_demands, label='Demand', color='blue', alpha=0.7)
        ax1.plot(test_indices, test_actual_allocations, label='Allocation', color='red', alpha=0.7)
        ax1.fill_between(test_indices, test_demands, test_actual_allocations, alpha=0.3, color='gray')
        ax1.set_xlabel('Timestep')
        ax1.set_ylabel('Resource Units')
        ax1.set_title('Test Set: Demand vs Allocation')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Cumulative rewards comparison
        train_cumulative = np.cumsum(train_individual_rewards)
        test_cumulative = np.cumsum(test_individual_rewards)
        
        ax2.plot(range(len(train_cumulative)), train_cumulative, label='Train', color='green', alpha=0.8)
        ax2.plot(range(len(test_cumulative)), test_cumulative, label='Test', color='orange', alpha=0.8)
        ax2.set_xlabel('Timestep')
        ax2.set_ylabel('Cumulative Reward')
        ax2.set_title('Cumulative Reward: Train vs Test')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Training evolution
        ax3.plot(selector._best_rewards, label='Best Reward', color='purple')
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Total Reward')
        ax3.set_title('Training Evolution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Allocation distribution comparison
        actions_list = sorted(set(list(train_action_distribution.keys()) + list(test_action_distribution.keys())))
        train_counts = [train_action_distribution.get(action, 0) for action in actions_list]
        test_counts = [test_action_distribution.get(action, 0) for action in actions_list]
        
        x = np.arange(len(actions_list))
        width = 0.35
        
        ax4.bar(x - width/2, train_counts, width, label='Train', color='lightblue', alpha=0.7)
        ax4.bar(x + width/2, test_counts, width, label='Test', color='orange', alpha=0.7)
        ax4.set_xlabel('Allocation Level')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Allocation Distribution: Train vs Test')
        ax4.set_xticks(x)
        ax4.set_xticklabels(actions_list)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("\nMatplotlib not available - skipping plots")
    
    print(f"\n{selector}")


if __name__ == "__main__":
    main()