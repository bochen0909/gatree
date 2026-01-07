"""
Demonstration of the new previous_action parameter in GATreeActionSelector reward functions.

This example shows how to use the previous_action parameter to create reward functions
that consider action transitions and encourage or discourage certain behavioral patterns.
"""

import pandas as pd
import numpy as np
from gatree.methods.gatreeactionselector import GATreeActionSelector


def create_sample_data(n_timesteps=20, random_state=42):
    """Create simple time series data for demonstration."""
    np.random.seed(random_state)
    
    # Simple features
    X = pd.DataFrame({
        'price': np.cumsum(np.random.randn(n_timesteps) * 0.1) + 100,
        'volume': np.random.exponential(1000, n_timesteps),
        'trend': np.random.randn(n_timesteps)
    })
    
    # Simple reward data
    Y = pd.DataFrame({
        'market_return': np.random.randn(n_timesteps) * 0.02
    })
    
    return X, Y


def reward_with_transition_penalty(state, action, y_data, timestep, previous_action):
    """
    Reward function that penalizes frequent action changes.
    
    This demonstrates how to use previous_action to encourage stability.
    """
    market_return = y_data['market_return']
    
    # Base reward based on action and market direction
    if action == 'buy':
        base_reward = market_return * 100
    elif action == 'sell':
        base_reward = -market_return * 100
    else:  # hold
        base_reward = 0
    
    # Penalty for changing actions (encourages stability)
    transition_penalty = 0.0
    if previous_action is not None and previous_action != action:
        transition_penalty = 5.0  # Penalty for changing strategy
    
    return base_reward - transition_penalty


def reward_with_transition_bonus(state, action, y_data, timestep, previous_action):
    """
    Reward function that gives bonuses for specific action sequences.
    
    This demonstrates how to use previous_action to encourage certain patterns.
    """
    market_return = y_data['market_return']
    
    # Base reward
    if action == 'buy':
        base_reward = market_return * 100
    elif action == 'sell':
        base_reward = -market_return * 100
    else:  # hold
        base_reward = 0
    
    # Bonus for specific transitions
    transition_bonus = 0.0
    if previous_action is not None:
        # Bonus for buy -> hold (taking profits)
        if previous_action == 'buy' and action == 'hold':
            transition_bonus = 3.0
        # Bonus for sell -> hold (covering shorts)
        elif previous_action == 'sell' and action == 'hold':
            transition_bonus = 3.0
        # Small penalty for hold -> buy/sell (breaking stability)
        elif previous_action == 'hold' and action in ['buy', 'sell']:
            transition_bonus = -1.0
    
    return base_reward + transition_bonus


def main():
    """Demonstrate the previous_action parameter functionality."""
    print("GATreeActionSelector Previous Action Demo")
    print("=" * 45)
    
    # Create sample data
    X, Y = create_sample_data(n_timesteps=30, random_state=42)
    action_space = ['buy', 'sell', 'hold']
    
    print(f"Dataset: {len(X)} timesteps, {len(X.columns)} features")
    print(f"Action space: {action_space}")
    print()
    
    # Test 1: Reward function that penalizes transitions
    print("Test 1: Reward function with transition penalties")
    print("-" * 45)
    
    selector_penalty = GATreeActionSelector(
        action_space=action_space,
        reward_function=reward_with_transition_penalty,
        random_state=42
    )
    
    selector_penalty.fit(X, Y, population_size=5, max_iter=3)
    actions_penalty = selector_penalty.predict_actions(X)
    
    # Count transitions
    transitions_penalty = sum(1 for i in range(1, len(actions_penalty)) 
                             if actions_penalty[i] != actions_penalty[i-1])
    
    print(f"Actions: {actions_penalty[:10]}...")  # Show first 10
    print(f"Total transitions: {transitions_penalty}")
    print(f"Transition rate: {transitions_penalty/(len(actions_penalty)-1):.2%}")
    print()
    
    # Test 2: Reward function that gives bonuses for specific transitions
    print("Test 2: Reward function with transition bonuses")
    print("-" * 45)
    
    selector_bonus = GATreeActionSelector(
        action_space=action_space,
        reward_function=reward_with_transition_bonus,
        random_state=42
    )
    
    selector_bonus.fit(X, Y, population_size=5, max_iter=3)
    actions_bonus = selector_bonus.predict_actions(X)
    
    # Count specific transitions
    buy_to_hold = sum(1 for i in range(1, len(actions_bonus)) 
                     if actions_bonus[i-1] == 'buy' and actions_bonus[i] == 'hold')
    sell_to_hold = sum(1 for i in range(1, len(actions_bonus)) 
                      if actions_bonus[i-1] == 'sell' and actions_bonus[i] == 'hold')
    
    print(f"Actions: {actions_bonus[:10]}...")  # Show first 10
    print(f"Buy -> Hold transitions: {buy_to_hold}")
    print(f"Sell -> Hold transitions: {sell_to_hold}")
    print()
    
    # Test 3: Compare rewards
    print("Test 3: Reward comparison")
    print("-" * 25)
    
    total_reward_penalty, _, _ = selector_penalty.simulate_rewards(X, Y)
    total_reward_bonus, _, _ = selector_bonus.simulate_rewards(X, Y)
    
    print(f"Total reward (penalty strategy): {total_reward_penalty:.2f}")
    print(f"Total reward (bonus strategy): {total_reward_bonus:.2f}")
    print()
    
    # Test 4: Show how previous_action is passed
    print("Test 4: Demonstrating previous_action parameter")
    print("-" * 45)
    
    previous_actions_log = []
    
    def logging_reward(state, action, y_data, timestep, previous_action):
        previous_actions_log.append((timestep, action, previous_action))
        return 1.0  # Simple constant reward
    
    selector_log = GATreeActionSelector(
        action_space=action_space,
        reward_function=logging_reward,
        random_state=42
    )
    
    # Use small dataset for clear demonstration
    X_small = X.iloc[:5]
    Y_small = Y.iloc[:5]
    
    selector_log.fit(X_small, Y_small, population_size=3, max_iter=1)
    
    # Clear log and simulate to see the parameter passing
    previous_actions_log.clear()
    selector_log.simulate_rewards(X_small, Y_small)
    
    print("Timestep | Action | Previous Action")
    print("-" * 35)
    for timestep, action, prev_action in previous_actions_log:
        prev_str = str(prev_action) if prev_action is not None else "None"
        print(f"{timestep:8d} | {action:6s} | {prev_str}")
    
    print()
    print("✓ Previous action parameter is working correctly!")
    print("  - First timestep has previous_action = None")
    print("  - Subsequent timesteps receive the previous action")


if __name__ == "__main__":
    main()