"""
Example demonstrating GATreeContinuousActionSelector for continuous action selection.

This example shows how to use the continuous action selector for a simple trading scenario
where actions are continuous values between 0 and 1, representing the fraction of portfolio
to allocate to a risky asset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gatree.methods.gatreecontinuousactionselector import GATreeContinuousActionSelector


def create_synthetic_market_data(n_timesteps=100, random_state=42):
    """
    Create synthetic market data for demonstration.
    
    Args:
        n_timesteps (int): Number of time steps
        random_state (int): Random seed
        
    Returns:
        tuple: (X, Y) where X contains features and Y contains market data
    """
    np.random.seed(random_state)
    
    # Generate market features
    # Feature 1: Market volatility indicator
    volatility = np.random.uniform(0.1, 0.5, n_timesteps)
    
    # Feature 2: Trend indicator (momentum)
    trend = np.cumsum(np.random.randn(n_timesteps) * 0.1)
    trend = (trend - trend.min()) / (trend.max() - trend.min())  # Normalize to [0, 1]
    
    # Feature 3: Market sentiment (oscillating)
    sentiment = 0.5 + 0.3 * np.sin(np.linspace(0, 4*np.pi, n_timesteps)) + np.random.randn(n_timesteps) * 0.1
    sentiment = np.clip(sentiment, 0, 1)
    
    X = pd.DataFrame({
        'volatility': volatility,
        'trend': trend,
        'sentiment': sentiment
    })
    
    # Generate market returns and risk-free rate
    market_returns = np.random.randn(n_timesteps) * volatility + trend * 0.02
    risk_free_rate = np.full(n_timesteps, 0.001)  # 0.1% per period
    
    Y = pd.DataFrame({
        'market_return': market_returns,
        'risk_free_rate': risk_free_rate,
        'volatility': volatility
    })
    
    return X, Y


def portfolio_reward_function(state, action, y_data, timestep, previous_action):
    """
    Portfolio allocation reward function.
    
    Args:
        state (pd.Series): Current market state features
        action (float): Portfolio allocation (0 = all risk-free, 1 = all risky asset)
        y_data (pd.Series): Market data for this timestep
        timestep (int): Current timestep
        previous_action (float): Previous action (None for first timestep)
        
    Returns:
        float: Reward for this action
    """
    try:
        # Get market data
        market_return = float(y_data['market_return'])
        risk_free_rate = float(y_data['risk_free_rate'])
        volatility = float(y_data['volatility'])
        
        # Calculate portfolio return
        # action = 0: all in risk-free asset
        # action = 1: all in risky asset
        portfolio_return = action * market_return + (1 - action) * risk_free_rate
        
        # Calculate risk penalty (higher allocation to risky asset = higher risk)
        risk_penalty = 0.5 * (action ** 2) * volatility
        
        # Transaction cost penalty for changing allocation
        transaction_cost = 0.0
        if previous_action is not None:
            allocation_change = abs(action - previous_action)
            transaction_cost = 0.01 * allocation_change  # 1% cost per unit change
        
        # Total reward = return - risk penalty - transaction costs
        reward = portfolio_return - risk_penalty - transaction_cost
        
        return reward
        
    except Exception as e:
        print(f"Error in reward function at timestep {timestep}: {e}")
        return -0.1  # Small penalty for errors


def global_portfolio_reward(rewards, actions, X, Y):
    """
    Global reward function that considers overall portfolio performance.
    
    Args:
        rewards (list): Individual timestep rewards
        actions (list): Action sequence
        X (pd.DataFrame): Feature data
        Y (pd.DataFrame): Market data
        
    Returns:
        float: Global reward component
    """
    try:
        # Calculate Sharpe ratio as global reward
        if len(rewards) < 2:
            return 0.0
        
        returns_array = np.array(rewards)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        
        if std_return == 0:
            return mean_return * 10  # High reward for consistent returns
        
        sharpe_ratio = mean_return / std_return
        
        # Scale the Sharpe ratio
        return sharpe_ratio * 0.1
        
    except Exception as e:
        print(f"Error in global reward function: {e}")
        return 0.0


def main():
    """Main example function."""
    print("GATree Continuous Action Selector Example")
    print("=" * 50)
    
    # Create synthetic market data
    print("Creating synthetic market data...")
    X, Y = create_synthetic_market_data(n_timesteps=200, random_state=42)
    
    print(f"Data shape: X={X.shape}, Y={Y.shape}")
    print(f"Features: {list(X.columns)}")
    print(f"Market data: {list(Y.columns)}")
    
    # Split data into train and test
    train_size = int(0.7 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    Y_train, Y_test = Y[:train_size], Y[train_size:]
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Create continuous action selector
    print("\nCreating GATreeContinuousActionSelector...")
    selector = GATreeContinuousActionSelector(
        reward_function=portfolio_reward_function,
        action_bounds=(0, 1),  # 0 = all risk-free, 1 = all risky
        max_depth=6,
        discount_factor=0.99,  # Slight preference for immediate rewards
        discount_direction='forward',
        global_reward_function=global_portfolio_reward,
        global_reward_weight=0.2,  # 20% weight on global reward
        n_action_bins=50,  # Fine-grained continuous actions
        random_state=42
    )
    
    # Fit the model
    print("Training the model...")
    
    def progress_callback(generation, best_fitness, avg_fitness, best_reward):
        if generation % 20 == 0:
            print(f"Generation {generation:3d}: Best Reward = {best_reward:8.4f}, "
                  f"Best Fitness = {best_fitness:8.4f}, Avg Fitness = {avg_fitness:8.4f}")
    
    selector.fit(
        X_train, Y_train,
        population_size=50,
        max_iter=100,
        mutation_probability=0.2,
        elite_size=5,
        selection_tournament_size=3,
        progress_callback=progress_callback,
        early_stopping=True,
        patience=20,
        min_delta=1e-4
    )
    
    print(f"\nTraining completed!")
    print(f"Final tree depth: {selector._tree.max_depth()}")
    print(f"Final tree size: {selector._tree.size()}")
    
    # Make predictions on test data
    print("\nMaking predictions on test data...")
    test_actions = selector.predict_actions(X_test)
    
    # Get action statistics
    action_stats = selector.get_action_statistics(X_test)
    print(f"\nTest Action Statistics:")
    for key, value in action_stats.items():
        print(f"  {key}: {value:.4f}")
    
    # Simulate rewards on test data
    total_reward, individual_rewards, _ = selector.simulate_rewards(X_test, Y_test)
    print(f"\nTest Performance:")
    print(f"  Total Reward: {total_reward:.4f}")
    print(f"  Average Reward per Timestep: {np.mean(individual_rewards):.4f}")
    print(f"  Reward Std: {np.std(individual_rewards):.4f}")
    
    # Compare with simple strategies
    print("\nComparing with baseline strategies...")
    
    # Strategy 1: Always 50% allocation
    baseline_actions_50 = [0.5] * len(X_test)
    baseline_rewards_50 = []
    prev_action = None
    for t, action in enumerate(baseline_actions_50):
        reward = portfolio_reward_function(X_test.iloc[t], action, Y_test.iloc[t], t, prev_action)
        baseline_rewards_50.append(reward)
        prev_action = action
    
    # Strategy 2: Always 20% allocation (conservative)
    baseline_actions_20 = [0.2] * len(X_test)
    baseline_rewards_20 = []
    prev_action = None
    for t, action in enumerate(baseline_actions_20):
        reward = portfolio_reward_function(X_test.iloc[t], action, Y_test.iloc[t], t, prev_action)
        baseline_rewards_20.append(reward)
        prev_action = action
    
    print(f"  50% Allocation Strategy: {np.mean(baseline_rewards_50):.4f} ± {np.std(baseline_rewards_50):.4f}")
    print(f"  20% Allocation Strategy: {np.mean(baseline_rewards_20):.4f} ± {np.std(baseline_rewards_20):.4f}")
    print(f"  GATree Strategy: {np.mean(individual_rewards):.4f} ± {np.std(individual_rewards):.4f}")
    
    # Plot results
    print("\nGenerating plots...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Training evolution
    ax1.plot(selector._best_rewards, label='Best Reward', color='green')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Training Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Action sequence
    ax2.plot(test_actions, label='GATree Actions', color='blue', linewidth=2)
    ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='50% Allocation')
    ax2.axhline(y=0.2, color='orange', linestyle='--', alpha=0.7, label='20% Allocation')
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Portfolio Allocation')
    ax2.set_title('Predicted Action Sequence (Test Data)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 1.05)
    
    # Plot 3: Reward comparison
    timesteps = range(len(individual_rewards))
    ax3.plot(timesteps, individual_rewards, label='GATree', alpha=0.8)
    ax3.plot(timesteps, baseline_rewards_50, label='50% Allocation', alpha=0.8)
    ax3.plot(timesteps, baseline_rewards_20, label='20% Allocation', alpha=0.8)
    ax3.set_xlabel('Timestep')
    ax3.set_ylabel('Reward')
    ax3.set_title('Reward Comparison (Test Data)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Market features
    ax4.plot(X_test.index, X_test['volatility'], label='Volatility', alpha=0.7)
    ax4.plot(X_test.index, X_test['trend'], label='Trend', alpha=0.7)
    ax4.plot(X_test.index, X_test['sentiment'], label='Sentiment', alpha=0.7)
    ax4.set_xlabel('Timestep')
    ax4.set_ylabel('Feature Value')
    ax4.set_title('Market Features (Test Data)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print tree structure (simplified)
    print(f"\nTrained Tree Structure:")
    print(f"  Depth: {selector._tree.max_depth()}")
    print(f"  Size: {selector._tree.size()}")
    print(f"  Action Bounds: {selector.action_bounds}")
    print(f"  Action Bins: {selector.n_action_bins}")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()