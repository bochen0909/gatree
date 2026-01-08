"""
Improved example demonstrating GATreeContinuousActionSelector with better action diversity.

This example creates market conditions that reward different allocation strategies
in different market regimes, encouraging the tree to learn diverse actions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gatree.methods.gatreecontinuousactionselector import GATreeContinuousActionSelector


def create_diverse_market_data(n_timesteps=200, random_state=42):
    """
    Create market data with distinct regimes that reward different strategies.
    """
    np.random.seed(random_state)
    
    # Create distinct market regimes
    regime_length = n_timesteps // 4
    
    # Regime 1: Bull market (high trend, low volatility) - should favor high allocation
    bull_trend = np.linspace(0.8, 0.9, regime_length)
    bull_volatility = np.full(regime_length, 0.1)
    bull_sentiment = np.linspace(0.7, 0.9, regime_length)
    
    # Regime 2: Bear market (low trend, high volatility) - should favor low allocation  
    bear_trend = np.linspace(0.1, 0.2, regime_length)
    bear_volatility = np.full(regime_length, 0.4)
    bear_sentiment = np.linspace(0.1, 0.3, regime_length)
    
    # Regime 3: Sideways market (medium trend, medium volatility) - should favor medium allocation
    sideways_trend = np.full(regime_length, 0.5) + np.random.randn(regime_length) * 0.05
    sideways_volatility = np.full(regime_length, 0.2)
    sideways_sentiment = np.full(regime_length, 0.5) + np.random.randn(regime_length) * 0.1
    
    # Regime 4: Recovery market (increasing trend, decreasing volatility) - should favor increasing allocation
    recovery_trend = np.linspace(0.3, 0.8, regime_length)
    recovery_volatility = np.linspace(0.3, 0.15, regime_length)
    recovery_sentiment = np.linspace(0.4, 0.8, regime_length)
    
    # Combine regimes
    trend = np.concatenate([bull_trend, bear_trend, sideways_trend, recovery_trend])
    volatility = np.concatenate([bull_volatility, bear_volatility, sideways_volatility, recovery_volatility])
    sentiment = np.concatenate([bull_sentiment, bear_sentiment, sideways_sentiment, recovery_sentiment])
    
    # Add some noise and ensure bounds
    trend = np.clip(trend + np.random.randn(n_timesteps) * 0.02, 0, 1)
    volatility = np.clip(volatility + np.random.randn(n_timesteps) * 0.02, 0.05, 0.5)
    sentiment = np.clip(sentiment + np.random.randn(n_timesteps) * 0.02, 0, 1)
    
    X = pd.DataFrame({
        'trend': trend,
        'volatility': volatility,
        'sentiment': sentiment
    })
    
    # Generate returns that strongly depend on market regime
    market_returns = np.zeros(n_timesteps)
    
    for i in range(n_timesteps):
        base_return = np.random.randn() * 0.01
        
        # Strong trend effect
        trend_effect = (trend[i] - 0.5) * 0.08  # Strong effect
        
        # Sentiment effect
        sentiment_effect = (sentiment[i] - 0.5) * 0.04
        
        # Volatility effect (random component)
        volatility_effect = np.random.randn() * volatility[i] * 1.5
        
        market_returns[i] = base_return + trend_effect + sentiment_effect + volatility_effect
    
    Y = pd.DataFrame({
        'market_return': market_returns,
        'risk_free_rate': np.full(n_timesteps, 0.002),  # 0.2% per period
        'volatility': volatility
    })
    
    return X, Y


def smart_portfolio_reward(state, action, y_data, timestep, previous_action):
    """
    Portfolio reward function that encourages different actions in different market conditions.
    """
    try:
        # Get market data
        market_return = float(y_data['market_return'])
        risk_free_rate = float(y_data['risk_free_rate'])
        volatility = float(y_data['volatility'])
        
        # Get state features
        trend = float(state['trend'])
        sentiment = float(state['sentiment'])
        
        # Calculate portfolio return
        portfolio_return = action * market_return + (1 - action) * risk_free_rate
        
        # Adaptive risk penalty based on market conditions
        # In good markets (high trend, high sentiment), lower risk penalty
        # In bad markets, higher risk penalty
        market_quality = (trend + sentiment) / 2.0
        base_risk_penalty = 0.1 * (1.0 - market_quality)  # Lower penalty in good markets
        risk_penalty = base_risk_penalty * (action ** 2) * volatility
        
        # Opportunity cost: penalty for being too conservative in good markets
        if market_quality > 0.7 and action < 0.3:
            opportunity_cost = 0.05 * (0.3 - action)  # Penalty for low allocation in good markets
        else:
            opportunity_cost = 0.0
        
        # Overexposure penalty: penalty for being too aggressive in bad markets
        if market_quality < 0.3 and action > 0.7:
            overexposure_penalty = 0.1 * (action - 0.7)  # Penalty for high allocation in bad markets
        else:
            overexposure_penalty = 0.0
        
        # Transaction cost (smaller than before)
        transaction_cost = 0.0
        if previous_action is not None:
            allocation_change = abs(action - previous_action)
            transaction_cost = 0.005 * allocation_change  # 0.5% cost per unit change
        
        # Market timing bonus: reward for matching allocation to market conditions
        optimal_allocation = market_quality  # In good markets, higher allocation is better
        timing_bonus = 0.02 * (1.0 - abs(action - optimal_allocation))
        
        # Total reward
        reward = (portfolio_return + timing_bonus 
                 - risk_penalty - opportunity_cost - overexposure_penalty - transaction_cost)
        
        return reward
        
    except Exception as e:
        print(f"Error in reward function at timestep {timestep}: {e}")
        return -0.1


def regime_global_reward(rewards, actions, X, Y):
    """
    Global reward that considers regime-appropriate behavior.
    """
    try:
        if len(rewards) < 4:
            return 0.0
        
        # Calculate regime-specific performance
        n_timesteps = len(rewards)
        regime_length = n_timesteps // 4
        
        regime_rewards = []
        regime_actions = []
        
        for i in range(4):
            start_idx = i * regime_length
            end_idx = (i + 1) * regime_length if i < 3 else n_timesteps
            
            regime_reward = np.mean(rewards[start_idx:end_idx])
            regime_action = np.mean(actions[start_idx:end_idx])
            
            regime_rewards.append(regime_reward)
            regime_actions.append(regime_action)
        
        # Bonus for appropriate regime behavior
        # Bull market (regime 0): should have high allocation
        # Bear market (regime 1): should have low allocation
        # Sideways (regime 2): should have medium allocation
        # Recovery (regime 3): should have increasing allocation
        
        regime_bonus = 0.0
        
        # Bull market bonus
        if regime_actions[0] > 0.6:  # High allocation in bull market
            regime_bonus += 0.1
        
        # Bear market bonus
        if regime_actions[1] < 0.3:  # Low allocation in bear market
            regime_bonus += 0.1
        
        # Sideways market bonus
        if 0.3 <= regime_actions[2] <= 0.7:  # Medium allocation in sideways market
            regime_bonus += 0.05
        
        # Recovery market bonus (increasing allocation)
        recovery_start = 3 * regime_length
        recovery_actions = actions[recovery_start:]
        if len(recovery_actions) > 10:
            recovery_trend = np.polyfit(range(len(recovery_actions)), recovery_actions, 1)[0]
            if recovery_trend > 0:  # Increasing allocation
                regime_bonus += 0.1
        
        return regime_bonus
        
    except Exception as e:
        print(f"Error in global reward function: {e}")
        return 0.0


def main():
    """Main example function."""
    print("Improved GATree Continuous Action Selector Example")
    print("=" * 55)
    
    # Create diverse market data
    print("Creating diverse market data with distinct regimes...")
    X, Y = create_diverse_market_data(n_timesteps=200, random_state=42)
    
    print(f"Data shape: X={X.shape}, Y={Y.shape}")
    print(f"Features: {list(X.columns)}")
    
    # Show regime characteristics
    regime_length = len(X) // 4
    regimes = ['Bull', 'Bear', 'Sideways', 'Recovery']
    
    print("\nMarket Regime Characteristics:")
    for i, regime_name in enumerate(regimes):
        start_idx = i * regime_length
        end_idx = (i + 1) * regime_length if i < 3 else len(X)
        
        regime_data = X.iloc[start_idx:end_idx]
        print(f"  {regime_name:8s}: trend={regime_data['trend'].mean():.2f}, "
              f"volatility={regime_data['volatility'].mean():.2f}, "
              f"sentiment={regime_data['sentiment'].mean():.2f}")
    
    # Split data
    train_size = int(0.7 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    Y_train, Y_test = Y[:train_size], Y[train_size:]
    
    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Create continuous action selector
    print("\nCreating GATreeContinuousActionSelector...")
    selector = GATreeContinuousActionSelector(
        reward_function=smart_portfolio_reward,
        action_bounds=(0, 1),
        max_depth=8,
        discount_factor=0.98,
        discount_direction='forward',
        global_reward_function=regime_global_reward,
        global_reward_weight=0.3,
        n_action_bins=100,
        random_state=42
    )
    
    # Fit the model
    print("Training the model...")
    
    def progress_callback(generation, best_fitness, avg_fitness, best_reward):
        if generation % 25 == 0:
            print(f"Generation {generation:3d}: Best Reward = {best_reward:8.4f}, "
                  f"Best Fitness = {best_fitness:8.4f}")
    
    selector.fit(
        X_train, Y_train,
        population_size=80,
        max_iter=150,
        mutation_probability=0.25,
        elite_size=8,
        selection_tournament_size=5,
        progress_callback=progress_callback,
        early_stopping=True,
        patience=25,
        min_delta=1e-5
    )
    
    print(f"\nTraining completed!")
    print(f"Final tree depth: {selector._tree.max_depth()}")
    print(f"Final tree size: {selector._tree.size()}")
    
    # Make predictions on test data
    print("\nMaking predictions on test data...")
    test_actions = selector.predict_actions(X_test)
    
    # Check action diversity
    unique_actions = len(set([round(a, 3) for a in test_actions]))  # Round to 3 decimals
    print(f"Number of unique actions (rounded): {unique_actions}")
    
    # Get action statistics
    action_stats = selector.get_action_statistics(X_test)
    print(f"\nTest Action Statistics:")
    for key, value in action_stats.items():
        print(f"  {key}: {value:.4f}")
    
    # Analyze actions by regime (if test data covers multiple regimes)
    if len(X_test) >= 40:  # Enough data to analyze
        test_regime_length = len(X_test) // 4
        print(f"\nAction Analysis by Market Regime (Test Data):")
        
        for i in range(4):
            start_idx = i * test_regime_length
            end_idx = (i + 1) * test_regime_length if i < 3 else len(X_test)
            
            if start_idx < len(test_actions):
                regime_actions = test_actions[start_idx:min(end_idx, len(test_actions))]
                regime_features = X_test.iloc[start_idx:min(end_idx, len(X_test))]
                
                print(f"  {regimes[i]:8s}: mean_action={np.mean(regime_actions):.3f}, "
                      f"std_action={np.std(regime_actions):.3f}, "
                      f"mean_trend={regime_features['trend'].mean():.2f}")
    
    # Simulate rewards on test data
    total_reward, individual_rewards, _ = selector.simulate_rewards(X_test, Y_test)
    print(f"\nTest Performance:")
    print(f"  Total Reward: {total_reward:.4f}")
    print(f"  Average Reward per Timestep: {np.mean(individual_rewards):.4f}")
    print(f"  Reward Std: {np.std(individual_rewards):.4f}")
    
    # Compare with baseline strategies
    print("\nComparing with baseline strategies...")
    
    # Strategy 1: Always 50% allocation
    baseline_50_rewards = []
    prev_action = None
    for t in range(len(X_test)):
        reward = smart_portfolio_reward(X_test.iloc[t], 0.5, Y_test.iloc[t], t, prev_action)
        baseline_50_rewards.append(reward)
        prev_action = 0.5
    
    # Strategy 2: Trend-following (action = trend)
    trend_following_rewards = []
    prev_action = None
    for t in range(len(X_test)):
        action = float(X_test.iloc[t]['trend'])
        reward = smart_portfolio_reward(X_test.iloc[t], action, Y_test.iloc[t], t, prev_action)
        trend_following_rewards.append(reward)
        prev_action = action
    
    print(f"  50% Allocation: {np.mean(baseline_50_rewards):.4f} ± {np.std(baseline_50_rewards):.4f}")
    print(f"  Trend Following: {np.mean(trend_following_rewards):.4f} ± {np.std(trend_following_rewards):.4f}")
    print(f"  GATree Strategy: {np.mean(individual_rewards):.4f} ± {np.std(individual_rewards):.4f}")
    
    # Plot results
    print("\nGenerating plots...")
    
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Training evolution
        ax1.plot(selector._best_rewards, label='Best Reward', color='green', linewidth=2)
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Total Reward')
        ax1.set_title('Training Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Action sequence with market features
        ax2_twin = ax2.twinx()
        
        # Plot actions
        ax2.plot(test_actions, label='GATree Actions', color='blue', linewidth=2)
        ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='50% Allocation')
        
        # Plot trend on secondary axis
        ax2_twin.plot(X_test.index, X_test['trend'], label='Market Trend', 
                     color='orange', alpha=0.7, linestyle=':')
        
        ax2.set_xlabel('Timestep')
        ax2.set_ylabel('Portfolio Allocation', color='blue')
        ax2_twin.set_ylabel('Market Trend', color='orange')
        ax2.set_title('Actions vs Market Trend (Test Data)')
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-0.05, 1.05)
        
        # Plot 3: Reward comparison
        timesteps = range(len(individual_rewards))
        ax3.plot(timesteps, individual_rewards, label='GATree', alpha=0.8, linewidth=2)
        ax3.plot(timesteps, baseline_50_rewards[:len(timesteps)], label='50% Allocation', alpha=0.8)
        ax3.plot(timesteps, trend_following_rewards[:len(timesteps)], label='Trend Following', alpha=0.8)
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
        plt.savefig('continuous_action_results.png', dpi=150, bbox_inches='tight')
        print("Plot saved as 'continuous_action_results.png'")
        
    except Exception as e:
        print(f"Plotting failed: {e}")
    
    print("\nExample completed successfully!")
    print(f"The GATree learned to use {unique_actions} different actions with std={action_stats['std']:.3f}")


if __name__ == "__main__":
    main()