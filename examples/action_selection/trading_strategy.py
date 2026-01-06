"""
Example of using GATreeActionSelector for a simple trading strategy.

This example demonstrates how to use the evolutionary decision tree action selector
to develop a trading strategy that maximizes profits over a time series of market data.
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


def generate_synthetic_market_data(n_timesteps=200, random_state=42):
    """
    Generate synthetic market data for demonstration.
    
    Returns:
        tuple: (features_df, price_df) where features contain technical indicators
               and price_df contains price information for reward calculation
    """
    np.random.seed(random_state)
    
    # Generate base price series with trend and noise
    base_trend = np.linspace(100, 120, n_timesteps)
    noise = np.random.normal(0, 2, n_timesteps)
    prices = base_trend + noise + np.sin(np.arange(n_timesteps) * 0.1) * 5
    
    # Ensure prices are positive
    prices = np.maximum(prices, 50)
    
    # Calculate technical indicators as features
    features = pd.DataFrame()
    
    # Price-based features
    features['price'] = prices
    features['price_change'] = np.concatenate([[0], np.diff(prices)])
    features['price_pct_change'] = features['price'].pct_change().fillna(0)
    
    # Moving averages
    features['ma_5'] = features['price'].rolling(window=5, min_periods=1).mean()
    features['ma_20'] = features['price'].rolling(window=20, min_periods=1).mean()
    features['ma_ratio'] = features['ma_5'] / features['ma_20']
    
    # Volatility (rolling standard deviation)
    features['volatility'] = features['price'].rolling(window=10, min_periods=1).std().fillna(0)
    
    # Momentum indicators
    features['momentum_5'] = features['price'] - features['price'].shift(5).fillna(features['price'])
    features['momentum_10'] = features['price'] - features['price'].shift(10).fillna(features['price'])
    
    # Volume (synthetic)
    features['volume'] = np.random.lognormal(10, 0.5, n_timesteps)
    
    # Price position within recent range
    features['price_position'] = features['price'].rolling(window=20, min_periods=1).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5
    ).fillna(0.5)
    
    # Create Y data for reward calculation (future prices)
    price_data = pd.DataFrame()
    price_data['current_price'] = prices
    price_data['next_price'] = np.concatenate([prices[1:], [prices[-1]]])
    price_data['price_change'] = price_data['next_price'] - price_data['current_price']
    
    return features, price_data


def trading_reward_function(state, action, price_data, timestep):
    """
    Calculate reward for a trading action.
    
    Args:
        state: Current market state (feature vector)
        action: Trading action ('buy', 'sell', 'hold')
        price_data: Price information for reward calculation
        timestep: Current timestep
    
    Returns:
        float: Reward for the action
    """
    current_price = price_data['current_price']
    next_price = price_data['next_price']
    price_change = price_data['price_change']
    
    # Transaction cost
    transaction_cost = 0.001 * current_price  # 0.1% of price
    
    if action == 'buy':
        # Reward is the price increase minus transaction cost
        reward = price_change - transaction_cost
    elif action == 'sell':
        # Reward is the negative price change (profit from short selling) minus cost
        reward = -price_change - transaction_cost
    else:  # hold
        # Small penalty for inaction to encourage trading when profitable
        reward = -0.0001 * current_price
    
    return reward


def main():
    """
    Main function to demonstrate GATreeActionSelector for trading.
    """
    print("GATree Action Selector Example - Trading Strategy")
    print("=" * 55)
    
    # Generate synthetic market data
    print("Generating synthetic market data...")
    features, price_data = generate_synthetic_market_data(n_timesteps=300, random_state=42)
    
    print(f"Dataset shape: {features.shape}")
    print(f"Price range: [{price_data['current_price'].min():.2f}, {price_data['current_price'].max():.2f}]")
    print(f"Average price change: {price_data['price_change'].mean():.4f}")
    
    # Split data into train/test
    train_size = int(0.7 * len(features))
    
    X_train = features.iloc[:train_size].copy()
    Y_train = price_data.iloc[:train_size].copy()
    X_test = features.iloc[train_size:].copy()
    Y_test = price_data.iloc[train_size:].copy()
    
    print(f"Training period: {train_size} timesteps")
    print(f"Testing period: {len(X_test)} timesteps")
    
    # Scale features for better performance
    scaler = StandardScaler()
    feature_cols = ['price_change', 'price_pct_change', 'ma_ratio', 'volatility', 
                   'momentum_5', 'momentum_10', 'volume', 'price_position']
    
    X_train_scaled = X_train.copy()
    X_train_scaled[feature_cols] = scaler.fit_transform(X_train[feature_cols])
    
    X_test_scaled = X_test.copy()
    X_test_scaled[feature_cols] = scaler.transform(X_test[feature_cols])
    
    # Define action space
    action_space = ['buy', 'sell', 'hold']
    
    # Create and train the action selector
    print(f"\nTraining GATree Action Selector...")
    print(f"Action space: {action_space}")
    print("Parameters: population_size=30, max_iter=50, max_depth=6")
    
    selector = GATreeActionSelector(
        action_space=action_space,
        reward_function=trading_reward_function,
        max_depth=6,
        discount_factor=0.99,  # Slight discount for future rewards
        n_jobs=2,
        random_state=42
    )
    
    # Train with smaller parameters for demonstration
    selector.fit(
        X=X_train_scaled,
        Y=Y_train,
        population_size=30,
        max_iter=50,
        mutation_probability=0.2,
        elite_size=2,
        selection_tournament_size=3
    )
    
    # Evaluate the strategy on both train and test data
    print("\nEvaluating trading strategy...")
    
    # Evaluate on training data
    print("Evaluating on training data...")
    train_total_reward, train_individual_rewards, train_actions = selector.simulate_rewards(
        X_train_scaled, Y_train, trading_reward_function
    )
    train_action_distribution = selector.get_action_distribution(X_train_scaled)
    
    # Evaluate on test data
    print("Evaluating on test data...")
    test_total_reward, test_individual_rewards, test_actions = selector.simulate_rewards(
        X_test_scaled, Y_test, trading_reward_function
    )
    test_action_distribution = selector.get_action_distribution(X_test_scaled)
    
    print("\nTrading Results:")
    print("=" * 40)
    
    # Training results
    print("TRAINING SET:")
    print(f"  Total reward: {train_total_reward:.4f}")
    print(f"  Average reward per timestep: {train_total_reward / len(X_train):.4f}")
    print(f"  Number of timesteps: {len(X_train)}")
    
    # Test results
    print("TEST SET:")
    print(f"  Total reward: {test_total_reward:.4f}")
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

    print(f"\nAction Distribution Comparison:")
    print("Action\tTrain\tTest\tDifference")
    print("-" * 35)
    for action in ['buy', 'sell', 'hold']:
        train_count = train_action_distribution.get(action, 0)
        test_count = test_action_distribution.get(action, 0)
        train_pct = (train_count / len(train_actions)) * 100
        test_pct = (test_count / len(test_actions)) * 100
        diff = test_pct - train_pct
        print(f"{action}\t{train_pct:.1f}%\t{test_pct:.1f}%\t{diff:+.1f}%")
    
    # Show sample trading decisions from test set
    print(f"\nSample Test Set Trading Decisions:")
    print("-" * 50)
    print("Timestep\tPrice\tAction\tReward\tNext Price")
    for i in range(min(10, len(test_actions))):
        idx = train_size + i
        current_price = price_data.iloc[idx]['current_price']
        next_price = price_data.iloc[idx]['next_price']
        action = test_actions[i]
        reward = test_individual_rewards[i]
        print(f"{i+1}\t\t{current_price:.2f}\t{action}\t{reward:.4f}\t{next_price:.2f}")
    
    # Compare with simple strategies on both datasets
    print(f"\nStrategy Comparison:")
    print("-" * 50)
    
    def evaluate_baseline_strategy(X_data, Y_data, action, strategy_name):
        rewards = []
        for i in range(len(X_data)):
            reward = trading_reward_function(
                X_data.iloc[i], action, Y_data.iloc[i], i
            )
            rewards.append(reward)
        return sum(rewards)
    
    # Training baselines
    train_buy_total = evaluate_baseline_strategy(X_train, Y_train, 'buy', "Always Buy")
    train_hold_total = evaluate_baseline_strategy(X_train, Y_train, 'hold', "Always Hold")
    train_sell_total = evaluate_baseline_strategy(X_train, Y_train, 'sell', "Always Sell")
    
    # Test baselines
    test_buy_total = evaluate_baseline_strategy(X_test, Y_test, 'buy', "Always Buy")
    test_hold_total = evaluate_baseline_strategy(X_test, Y_test, 'hold', "Always Hold")
    test_sell_total = evaluate_baseline_strategy(X_test, Y_test, 'sell', "Always Sell")
    
    print("Strategy\t\tTrain\t\tTest\t\tGeneralization")
    print("-" * 65)
    print(f"GATree\t\t\t{train_total_reward:.4f}\t\t{test_total_reward:.4f}\t\t{test_total_reward/train_total_reward:.3f}")
    print(f"Always Buy\t\t{train_buy_total:.4f}\t\t{test_buy_total:.4f}\t\t{test_buy_total/train_buy_total:.3f}")
    print(f"Always Hold\t\t{train_hold_total:.4f}\t\t{test_hold_total:.4f}\t\t{test_hold_total/train_hold_total:.3f}")
    print(f"Always Sell\t\t{train_sell_total:.4f}\t\t{test_sell_total:.4f}\t\t{test_sell_total/train_sell_total:.3f}")
    
    # Best baseline comparison
    best_train_baseline = max(train_buy_total, train_hold_total, train_sell_total)
    best_test_baseline = max(test_buy_total, test_hold_total, test_sell_total)
    
    train_improvement = ((train_total_reward - best_train_baseline) / abs(best_train_baseline)) * 100 if best_train_baseline != 0 else 0
    test_improvement = ((test_total_reward - best_test_baseline) / abs(best_test_baseline)) * 100 if best_test_baseline != 0 else 0
    
    print(f"\nImprovement vs best baseline:")
    print(f"  Training: {train_improvement:.1f}%")
    print(f"  Test: {test_improvement:.1f}%")
    
    # Show training evolution
    print(f"\nTraining Evolution:")
    print(f"Initial best reward: {selector._best_rewards[0]:.4f}")
    print(f"Final best reward: {selector._best_rewards[-1]:.4f}")
    print(f"Improvement: {selector._best_rewards[-1] - selector._best_rewards[0]:.4f}")
    
    # Tree information
    print(f"\nFinal Tree Information:")
    print(f"Tree depth: {selector._tree.max_depth()}")
    print(f"Tree size (nodes): {selector._tree.size()}")
    print(f"Number of leaves: {len(selector._tree.get_leaves())}")
    
    # Plot results if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Price and actions (Test Set)
        test_indices = range(len(X_test))
        test_prices = Y_test['current_price'].values
        
        ax1.plot(test_indices, test_prices, label='Price', color='black', alpha=0.7)
        
        # Color-code actions
        buy_indices = [i for i, a in enumerate(test_actions) if a == 'buy']
        sell_indices = [i for i, a in enumerate(test_actions) if a == 'sell']
        hold_indices = [i for i, a in enumerate(test_actions) if a == 'hold']
        
        if buy_indices:
            ax1.scatter([test_indices[i] for i in buy_indices], 
                       [test_prices[i] for i in buy_indices], 
                       color='green', label='Buy', alpha=0.7, s=30)
        if sell_indices:
            ax1.scatter([test_indices[i] for i in sell_indices], 
                       [test_prices[i] for i in sell_indices], 
                       color='red', label='Sell', alpha=0.7, s=30)
        if hold_indices:
            ax1.scatter([test_indices[i] for i in hold_indices], 
                       [test_prices[i] for i in hold_indices], 
                       color='gray', label='Hold', alpha=0.3, s=10)
        
        ax1.set_xlabel('Timestep')
        ax1.set_ylabel('Price')
        ax1.set_title('Test Set: Trading Actions vs Price')
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
        ax3.plot(selector._best_rewards, label='Best Reward', color='green')
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Total Reward')
        ax3.set_title('Training Evolution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Action distribution comparison
        actions_list = ['buy', 'sell', 'hold']
        train_counts = [train_action_distribution.get(action, 0) for action in actions_list]
        test_counts = [test_action_distribution.get(action, 0) for action in actions_list]
        
        x = np.arange(len(actions_list))
        width = 0.35
        
        ax4.bar(x - width/2, train_counts, width, label='Train', color='lightblue', alpha=0.7)
        ax4.bar(x + width/2, test_counts, width, label='Test', color='orange', alpha=0.7)
        ax4.set_xlabel('Action')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Action Distribution: Train vs Test')
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