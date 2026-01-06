# GATree Action Selection Examples

This directory contains examples demonstrating how to use `GATreeActionSelector` for sequential decision-making tasks in time series data.

## Overview

The `GATreeActionSelector` is an evolutionary decision tree that learns to make optimal actions at each time step to maximize cumulative rewards over a time series. Unlike traditional supervised learning, there are no fixed targets - the fitness is determined by the total reward achieved by the action sequence.

## Key Concepts

### **Sequential Decision Making**
- Each row in the time series represents a state where an action must be taken
- Actions are selected based on current features using evolved decision trees
- The goal is to maximize cumulative rewards across the entire sequence

### **Reward-Based Learning**
- No traditional target variable (y) - instead uses reward signals
- Fitness function evaluates entire action sequences, not individual predictions
- Genetic algorithm evolves trees that maximize total rewards

### **Action Space**
- Discrete set of possible actions (e.g., ['buy', 'sell', 'hold'])
- Tree leaves contain action indices that map to actual actions
- Actions can be categorical (strings) or numerical values

## Architecture

```python
GATreeActionSelector(
    action_space=['action1', 'action2', 'action3'],
    reward_function=custom_reward_function,
    max_depth=6,
    discount_factor=0.95,
    n_jobs=2
)
```

### **Key Components:**

1. **Action Space**: List of possible actions
2. **Reward Function**: `reward_function(state, action, y_data, timestep) -> reward`
3. **Discount Factor**: Weight for future rewards (0.0 to 1.0)
4. **Genetic Algorithm**: Evolves tree structures to maximize rewards

## Examples

### 1. Trading Strategy (`trading_strategy.py`)

**Problem**: Develop a trading strategy that maximizes profits over market data.

**Features (X)**:
- Price, price changes, moving averages
- Technical indicators (momentum, volatility)
- Market position metrics

**Reward Data (Y)**:
- Current and next prices for profit calculation
- Transaction costs

**Actions**: `['buy', 'sell', 'hold']`

**Reward Function**:
```python
def trading_reward_function(state, action, price_data, timestep):
    current_price = price_data['current_price']
    next_price = price_data['next_price']
    transaction_cost = 0.001 * current_price
    
    if action == 'buy':
        return (next_price - current_price) - transaction_cost
    elif action == 'sell':
        return (current_price - next_price) - transaction_cost
    else:  # hold
        return -0.0001 * current_price  # Small inaction penalty
```

**Usage**:
```bash
python examples/action_selection/trading_strategy.py
```

### 2. Resource Allocation (`resource_allocation.py`)

**Problem**: Optimize resource allocation decisions based on demand patterns and costs.

**Features (X)**:
- Current demand, demand trends, moving averages
- Time-based features (hour, day, weekend)
- System capacity and utilization metrics
- Resource availability

**Reward Data (Y)**:
- Unit costs, satisfaction values
- Penalty rates for over/under allocation

**Actions**: `[0, 1, 2, 3, 4, 5]` (resource units to allocate)

**Reward Function**:
```python
def resource_allocation_reward(state, action, cost_data, timestep):
    allocation = action
    demand = cost_data['actual_demand']
    unit_cost = cost_data['unit_cost']
    
    # Satisfaction reward
    satisfaction = min(allocation, demand) / demand
    satisfaction_reward = satisfaction * cost_data['satisfaction_value']
    
    # Costs and penalties
    allocation_cost = allocation * unit_cost
    waste_penalty = max(0, allocation - demand) * cost_data['overallocation_penalty']
    shortage_penalty = max(0, demand - allocation) * cost_data['underallocation_penalty']
    
    return satisfaction_reward - allocation_cost - waste_penalty - shortage_penalty
```

**Usage**:
```bash
python examples/action_selection/resource_allocation.py
```

## How It Works

### 1. **Tree Structure**
- **Internal Nodes**: Binary splits on features (`if feature > threshold`)
- **Leaf Nodes**: Contain action indices (0, 1, 2, ..., len(action_space)-1)
- **Prediction**: Tree traversal returns action index, mapped to actual action

### 2. **Fitness Evaluation**
```python
def default_fitness_function(root, X, Y, action_space, reward_function, discount_factor):
    total_reward = 0.0
    
    for t in range(len(X)):
        action_idx = root.predict_one(X.iloc[t])
        action = action_space[action_idx]
        reward = reward_function(X.iloc[t], action, Y.iloc[t], t)
        total_reward += reward * (discount_factor ** t)
    
    complexity_penalty = 0.001 * root.size()
    return -total_reward + complexity_penalty  # Negative for minimization
```

### 3. **Genetic Evolution**
- **Population**: Multiple trees with different action strategies
- **Selection**: Tournament selection based on total rewards
- **Crossover**: Swap subtrees between high-performing trees
- **Mutation**: Modify tree structure (features, thresholds, actions)
- **Elitism**: Preserve best strategies across generations

### 4. **Training Process**
```python
# 1. Initialize random population of trees
# 2. For each generation:
#    a. Evaluate each tree on full time series
#    b. Calculate total discounted rewards
#    c. Select parents based on performance
#    d. Create offspring via crossover and mutation
#    e. Replace population with elites + best offspring
# 3. Return best tree from final generation
```

## Key Parameters

### **Constructor Parameters**:
- `action_space`: List of possible actions
- `reward_function`: Function to calculate rewards
- `max_depth`: Maximum tree depth (controls complexity)
- `discount_factor`: Future reward discount (0.0 = only immediate, 1.0 = no discount)
- `n_jobs`: Parallel processing cores

### **Training Parameters**:
- `population_size`: Number of trees per generation (larger = more exploration)
- `max_iter`: Number of evolutionary generations
- `mutation_probability`: Chance of mutating offspring (0.1-0.3 typical)
- `elite_size`: Number of best trees preserved each generation
- `selection_tournament_size`: Tournament size for parent selection

## Performance Tips

### **Feature Engineering**:
- Include relevant state information for decision making
- Add time-based features (hour, day, seasonality)
- Create derived features (moving averages, ratios, trends)
- Scale features for consistent tree splits

### **Reward Function Design**:
- Balance immediate vs. long-term rewards
- Include appropriate penalties (transaction costs, waste)
- Consider action consistency (avoid excessive switching)
- Make rewards proportional to problem scale

### **Hyperparameter Tuning**:
- **Population Size**: 50-200 (larger for complex problems)
- **Generations**: 100-1000 (more for better convergence)
- **Tree Depth**: 5-10 (deeper for complex decision boundaries)
- **Discount Factor**: 0.9-1.0 (lower for short-term focus)

### **Training Strategies**:
- Use cross-validation on different time periods
- Start with smaller populations for quick prototyping
- Monitor fitness evolution to detect convergence
- Use parallel processing (`n_jobs > 1`) for speed

## Advanced Usage

### **Custom Reward Functions**:
```python
def multi_objective_reward(state, action, data, timestep):
    """Combine multiple objectives with weights."""
    profit = calculate_profit(state, action, data)
    risk = calculate_risk(state, action, data)
    sustainability = calculate_sustainability(state, action, data)
    
    return 0.6 * profit - 0.3 * risk + 0.1 * sustainability
```

### **Temporal Constraints**:
```python
def constrained_reward(state, action, data, timestep, action_history):
    """Add constraints on action sequences."""
    base_reward = calculate_base_reward(state, action, data)
    
    # Penalty for too many consecutive same actions
    if len(action_history) >= 3 and all(a == action for a in action_history[-3:]):
        base_reward -= 10
    
    return base_reward
```

### **Rolling Window Training**:
```python
# Train on sliding windows for non-stationary environments
for start in range(0, len(X) - window_size, step_size):
    X_window = X.iloc[start:start + window_size]
    Y_window = Y.iloc[start:start + window_size]
    selector.fit(X_window, Y_window, max_iter=50)
```

## Expected Output

Both examples will display:

1. **Dataset Information**: Size, feature ranges, time periods
2. **Training Progress**: Population evolution, fitness improvement
3. **Performance Metrics**: Total rewards, efficiency measures
4. **Action Analysis**: Distribution, sample decisions
5. **Strategy Comparison**: vs. simple baseline strategies
6. **Visualizations**: Action sequences, reward evolution (if matplotlib available)

## Use Cases

### **Financial Applications**:
- Algorithmic trading strategies
- Portfolio rebalancing
- Risk management decisions
- Market making strategies

### **Operations Research**:
- Resource allocation optimization
- Inventory management
- Scheduling decisions
- Supply chain optimization

### **Control Systems**:
- Dynamic system control
- Process optimization
- Energy management
- Traffic flow control

### **Gaming and AI**:
- Game strategy development
- Multi-agent coordination
- Reinforcement learning alternatives
- Sequential decision problems

## Requirements

- pandas
- numpy
- scikit-learn
- joblib
- matplotlib (optional, for plotting)

## Notes

- **Training Time**: Longer than supervised learning due to sequence evaluation
- **Reward Design**: Critical for success - poorly designed rewards lead to poor strategies
- **Stationarity**: Assumes reward function is stationary over time
- **Scalability**: Works best with discrete action spaces (< 20 actions)
- **Validation**: Use time-based splits, not random splits for evaluation

## Comparison with Reinforcement Learning

| Aspect | GATreeActionSelector | Traditional RL |
|--------|---------------------|----------------|
| **Environment** | Static (no state changes from actions) | Dynamic (actions change environment) |
| **Learning** | Population-based evolution | Value/policy iteration |
| **Exploration** | Genetic diversity | ε-greedy, exploration bonuses |
| **Interpretability** | Decision tree (highly interpretable) | Neural networks (black box) |
| **Sample Efficiency** | Moderate (population-based) | Can be sample efficient |
| **Parallelization** | Natural (evaluate population) | Limited (sequential learning) |
| **Convergence** | Global optimization potential | Local optima risk |

The GATreeActionSelector is particularly well-suited for problems where:
- Actions don't change the environment state
- Interpretability is important
- Historical data is available for full sequence evaluation
- Discrete action spaces are sufficient