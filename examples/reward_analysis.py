"""
Analyze the reward function to understand why the tree chooses extreme actions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def smart_portfolio_reward(state, action, y_data, timestep, previous_action):
    """Same reward function from improved example."""
    try:
        market_return = float(y_data['market_return'])
        risk_free_rate = float(y_data['risk_free_rate'])
        volatility = float(y_data['volatility'])
        
        trend = float(state['trend'])
        sentiment = float(state['sentiment'])
        
        portfolio_return = action * market_return + (1 - action) * risk_free_rate
        
        market_quality = (trend + sentiment) / 2.0
        base_risk_penalty = 0.1 * (1.0 - market_quality)
        risk_penalty = base_risk_penalty * (action ** 2) * volatility
        
        if market_quality > 0.7 and action < 0.3:
            opportunity_cost = 0.05 * (0.3 - action)
        else:
            opportunity_cost = 0.0
        
        if market_quality < 0.3 and action > 0.7:
            overexposure_penalty = 0.1 * (action - 0.7)
        else:
            overexposure_penalty = 0.0
        
        transaction_cost = 0.0
        if previous_action is not None:
            allocation_change = abs(action - previous_action)
            transaction_cost = 0.005 * allocation_change
        
        optimal_allocation = market_quality
        timing_bonus = 0.02 * (1.0 - abs(action - optimal_allocation))
        
        reward = (portfolio_return + timing_bonus 
                 - risk_penalty - opportunity_cost - overexposure_penalty - transaction_cost)
        
        return reward
        
    except Exception as e:
        return -0.1


def analyze_reward_surface():
    """Analyze how rewards vary with different actions and market conditions."""
    
    # Create test scenarios
    scenarios = [
        {'name': 'Bull Market', 'trend': 0.8, 'sentiment': 0.8, 'volatility': 0.1, 'market_return': 0.05},
        {'name': 'Bear Market', 'trend': 0.2, 'sentiment': 0.2, 'volatility': 0.4, 'market_return': -0.03},
        {'name': 'Sideways Market', 'trend': 0.5, 'sentiment': 0.5, 'volatility': 0.2, 'market_return': 0.01},
        {'name': 'High Volatility', 'trend': 0.6, 'sentiment': 0.4, 'volatility': 0.5, 'market_return': 0.02},
    ]
    
    actions = np.linspace(0, 1, 21)  # 0%, 5%, 10%, ..., 100%
    
    print("Reward Analysis for Different Market Scenarios")
    print("=" * 50)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, scenario in enumerate(scenarios):
        print(f"\n{scenario['name']}:")
        print(f"  Trend: {scenario['trend']:.1f}, Sentiment: {scenario['sentiment']:.1f}")
        print(f"  Volatility: {scenario['volatility']:.1f}, Market Return: {scenario['market_return']:.3f}")
        
        # Create state and market data
        state = pd.Series({
            'trend': scenario['trend'],
            'volatility': scenario['volatility'],
            'sentiment': scenario['sentiment']
        })
        
        y_data = pd.Series({
            'market_return': scenario['market_return'],
            'risk_free_rate': 0.002,
            'volatility': scenario['volatility']
        })
        
        rewards = []
        best_action = 0
        best_reward = float('-inf')
        
        print("  Action -> Reward:")
        for action in actions:
            reward = smart_portfolio_reward(state, action, y_data, 0, None)
            rewards.append(reward)
            
            if reward > best_reward:
                best_reward = reward
                best_action = action
            
            if action in [0.0, 0.2, 0.5, 0.8, 1.0]:  # Show key points
                print(f"    {action:4.1f} -> {reward:7.4f}")
        
        print(f"  Best Action: {best_action:.1f} (Reward: {best_reward:.4f})")
        
        # Plot
        axes[i].plot(actions, rewards, 'b-', linewidth=2)
        axes[i].axvline(x=best_action, color='red', linestyle='--', alpha=0.7, label=f'Best: {best_action:.1f}')
        axes[i].set_xlabel('Action (Portfolio Allocation)')
        axes[i].set_ylabel('Reward')
        axes[i].set_title(scenario['name'])
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig('reward_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\nReward surface plots saved as 'reward_analysis.png'")
    
    # Analyze reward components
    print(f"\n" + "="*50)
    print("Detailed Reward Component Analysis (Bull Market)")
    print("="*50)
    
    scenario = scenarios[0]  # Bull market
    state = pd.Series({
        'trend': scenario['trend'],
        'volatility': scenario['volatility'], 
        'sentiment': scenario['sentiment']
    })
    
    y_data = pd.Series({
        'market_return': scenario['market_return'],
        'risk_free_rate': 0.002,
        'volatility': scenario['volatility']
    })
    
    test_actions = [0.0, 0.2, 0.5, 0.8, 1.0]
    
    for action in test_actions:
        print(f"\nAction = {action:.1f}:")
        
        # Calculate components manually
        market_return = scenario['market_return']
        risk_free_rate = 0.002
        volatility = scenario['volatility']
        trend = scenario['trend']
        sentiment = scenario['sentiment']
        
        portfolio_return = action * market_return + (1 - action) * risk_free_rate
        
        market_quality = (trend + sentiment) / 2.0
        base_risk_penalty = 0.1 * (1.0 - market_quality)
        risk_penalty = base_risk_penalty * (action ** 2) * volatility
        
        if market_quality > 0.7 and action < 0.3:
            opportunity_cost = 0.05 * (0.3 - action)
        else:
            opportunity_cost = 0.0
        
        if market_quality < 0.3 and action > 0.7:
            overexposure_penalty = 0.1 * (action - 0.7)
        else:
            overexposure_penalty = 0.0
        
        optimal_allocation = market_quality
        timing_bonus = 0.02 * (1.0 - abs(action - optimal_allocation))
        
        total_reward = (portfolio_return + timing_bonus 
                       - risk_penalty - opportunity_cost - overexposure_penalty)
        
        print(f"  Portfolio Return:     {portfolio_return:7.4f}")
        print(f"  Timing Bonus:         {timing_bonus:7.4f}")
        print(f"  Risk Penalty:        -{risk_penalty:7.4f}")
        print(f"  Opportunity Cost:    -{opportunity_cost:7.4f}")
        print(f"  Overexposure Penalty:-{overexposure_penalty:7.4f}")
        print(f"  Total Reward:         {total_reward:7.4f}")
        print(f"  Market Quality:       {market_quality:7.4f}")
        print(f"  Optimal Allocation:   {optimal_allocation:7.4f}")


if __name__ == "__main__":
    analyze_reward_surface()