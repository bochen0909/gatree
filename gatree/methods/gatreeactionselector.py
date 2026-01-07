import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator

# GATree
from gatree.tree.node import Node
from gatree.ga.selection import Selection
from gatree.ga.crossover import Crossover
from gatree.ga.mutation import Mutation
from gatree.gatree import GATree


class GATreeActionSelector(GATree, BaseEstimator):
    """
    Evolutionary decision tree for sequential action selection in time series.
    
    This class evolves decision trees that can make optimal actions at each time step
    to maximize cumulative rewards over a time series. Unlike traditional supervised
    learning, there are no fixed targets - the fitness is determined by the total
    reward achieved by the action sequence.

    Args:
        action_space (list): List of possible actions (e.g., ['buy', 'sell', 'hold'])
        reward_function (callable): Function to calculate rewards
        max_depth (int, optional): Maximum depth of the tree
        discount_factor (float, optional): Future reward discount factor (default: 1.0)
        random (Random, optional): Random number generator
        n_jobs (int, optional): Number of jobs to run in parallel
        random_state (int, optional): Seed for reproducibility

    Attributes:
        action_space (list): List of possible actions
        reward_function (callable): Reward calculation function
        discount_factor (float): Discount factor for future rewards
        max_depth (int): Maximum depth of the tree
        random (Random): Random number generator
        X (pandas.DataFrame): Training time series features
        Y (pandas.DataFrame): Training reward calculation data
        att_indexes (numpy.ndarray): Array of attribute indexes
        att_values (dict): Dictionary of attribute values
        action_count (int): Number of possible actions
        n_jobs (int): Number of jobs to run in parallel
        random_state (int): Seed for reproducibility
        _tree (Node): The fitted tree
        _best_fitness (list): List of best fitness values for each iteration
        _avg_fitness (list): List of average fitness values for each iteration
        _best_rewards (list): List of best total rewards for each iteration
    """

    def __init__(self, action_space, reward_function, max_depth=None, discount_factor=1.0, 
                 random=None, n_jobs=1, random_state=None):
        """
        Initialize the Genetic Algorithm Action Selector.

        Args:
            action_space (list): List of possible actions
            reward_function (callable): Function(state, action, y_data, timestep) -> reward
            max_depth (int, optional): Maximum depth of the tree
            discount_factor (float, optional): Future reward discount factor
            random (Random, optional): Random number generator
            n_jobs (int, optional): Number of jobs to run in parallel
            random_state (int, optional): Seed for reproducibility
        """
        super().__init__(max_depth, random, None, n_jobs, random_state)  # fitness_function set later
        
        self.action_space = action_space
        self.reward_function = reward_function
        self.discount_factor = discount_factor
        self.action_count = len(action_space)
        self._best_rewards = []
        
        # Set the fitness function to our reward-based function
        self.fitness_function = self.default_fitness_function

    @staticmethod
    def default_fitness_function(root, X, Y, action_space, reward_function, discount_factor=1.0, **kwargs):
        """
        Default fitness function for action selection based on cumulative rewards.
        
        Simulates the action sequence determined by the tree and calculates
        the total discounted reward minus a complexity penalty.

        Args:
            root (Node): Root node of the tree
            X (pandas.DataFrame): Time series features
            Y (pandas.DataFrame): Reward calculation data
            action_space (list): List of possible actions
            reward_function (callable): Reward calculation function
            discount_factor (float): Discount factor for future rewards

        Returns:
            float: Fitness value (lower is better, so we negate rewards)
        """
        # Check if tree has any predictions (for test compatibility)
        if hasattr(root, 'y_pred') and len(root.y_pred) == 0:
            return float('inf')
        
        total_reward = 0.0
        
        # Simulate action sequence and calculate rewards
        for t in range(len(X)):
            try:
                # Get action index from tree prediction
                action_idx = int(root.predict_one(X.iloc[t], train=False))
                
                # Ensure action index is valid
                if action_idx < 0 or action_idx >= len(action_space):
                    action_idx = 0  # Default to first action
                
                # Get actual action
                action = action_space[action_idx]
                
                # Calculate reward for this timestep
                reward = reward_function(X.iloc[t], action, Y.iloc[t], t)
                
                # Apply discount factor
                discounted_reward = reward * (discount_factor ** t)
                total_reward += discounted_reward
                
            except Exception as e:
                # Handle any errors in reward calculation
                print(f"Error calculating reward at timestep {t}: {e}")
                # Use a smaller penalty to avoid dominating the fitness
                total_reward -= 10  # Smaller penalty for errors
        
        # Add complexity penalty (encourage simpler trees)
        complexity_penalty = 0.001 * root.size()
        
        # Return negative reward (since GA minimizes fitness)
        fitness = -total_reward + complexity_penalty
        
        return fitness

    def fit(self, X, Y, population_size=100, max_iter=1000, mutation_probability=0.15, 
            elite_size=2, selection_tournament_size=3, fitness_function_kwargs={}):
        """
        Fit the action selector to time series data.

        Args:
            X (pandas.DataFrame): Time series features (rows = timesteps, cols = features)
            Y (pandas.DataFrame): Additional data for reward calculation
            population_size (int, optional): Size of the population
            max_iter (int, optional): Maximum number of iterations
            mutation_probability (float, optional): Probability of mutation
            elite_size (int, optional): Number of elite trees
            selection_tournament_size (int, optional): Tournament size for selection
            fitness_function_kwargs (dict, optional): Additional kwargs for fitness function

        Returns:
            self: The fitted action selector
        """
        self.X = X
        self.Y = Y
        self.att_indexes = np.arange(X.shape[1])
        
        # Create split thresholds for features (same as other GATree methods)
        self.att_values = {}
        for i in range(X.shape[1]):
            unique_vals = sorted(X.iloc[:, i].unique())
            if len(unique_vals) > 1:
                thresholds = [(min_val + max_val) / 2 for min_val, max_val in 
                             zip(unique_vals[:-1], unique_vals[1:])]
                self.att_values[i] = thresholds
            else:
                # Single unique value - create a dummy threshold
                self.att_values[i] = [unique_vals[0]]
        
        # Action space for leaf nodes (action indices)
        self.att_values[-1] = list(range(self.action_count))
        self.class_count = self.action_count
        
        # Add required parameters to fitness function kwargs
        fitness_function_kwargs.update({
            'action_space': self.action_space,
            'reward_function': self.reward_function,
            'discount_factor': self.discount_factor
        })

        # Generation of initial population
        node = Node()
        population = []
        for _ in range(population_size):
            tree = node.make_node(
                max_depth=self.max_depth, 
                random=self.random,
                att_indexes=self.att_indexes, 
                att_values=self.att_values, 
                class_count=self.class_count
            )
            population.append(tree)

        # Evolution loop
        for i in range(max_iter + 1):
            # Clear previous evaluation
            for tree in population:
                tree.clear_evaluation()

            # Evaluation of population
            population = Parallel(n_jobs=self.n_jobs)(
                delayed(GATreeActionSelector._predict_and_evaluate)(
                    tree, X, Y, self.fitness_function, True, **fitness_function_kwargs
                ) for tree in population
            )

            # Sort population by fitness (lower is better)
            population.sort(key=lambda x: x.fitness, reverse=False)

            # Log best and average fitness
            best_fitness = population[0].fitness
            avg_fitness = sum([tree.fitness for tree in population]) / len(population)
            
            self._best_fitness.append(best_fitness)
            self._avg_fitness.append(avg_fitness)
            
            # Calculate and log best reward 
            # Fitness = -total_reward + complexity_penalty
            # So: total_reward = -fitness + complexity_penalty
            if best_fitness != float('inf'):
                complexity_penalty = 0.001 * population[0].size()
                best_reward = -best_fitness + complexity_penalty
            else:
                best_reward = float('-inf')  # Invalid tree
            self._best_rewards.append(best_reward)

            if i != max_iter:
                # Elites
                elites = population[:elite_size]

                # Descendant generation
                descendant = []
                for _ in range(0, len(population), 2):
                    # Tournament selection
                    tree1, tree2 = Selection.selection(
                        population=population, 
                        selection_tournament_size=selection_tournament_size, 
                        random=self.random
                    )

                    # Crossover between selected trees
                    crossover1 = Crossover.crossover(tree1=tree1, tree2=tree2, random=self.random)
                    crossover2 = Crossover.crossover(tree1=tree2, tree2=tree1, random=self.random)

                    # Mutation of new trees
                    mutation1 = crossover1
                    mutation2 = crossover2
                    
                    if self.random.random() < mutation_probability:
                        mutation1 = Mutation.mutation(
                            root=crossover1, 
                            att_indexes=self.att_indexes,
                            att_values=self.att_values, 
                            class_count=self.class_count,
                            random=self.random
                        )
                    
                    if self.random.random() < mutation_probability:
                        mutation2 = Mutation.mutation(
                            root=crossover2, 
                            att_indexes=self.att_indexes,
                            att_values=self.att_values, 
                            class_count=self.class_count,
                            random=self.random
                        )

                    # Add new trees to descendant population
                    descendant.extend([mutation1, mutation2])

                # Elites + descendants
                descendant.sort(key=lambda x: x.fitness, reverse=False)
                descendant = elites + descendant[:population_size - elite_size]

                # Replace old population with new population
                population = descendant

        self._tree = population[0]
        return self

    @staticmethod
    def _predict_and_evaluate(tree, X, y, fitness_function, is_training=False, **fitness_function_kwargs):
        """
        Evaluate a tree on time series data (overrides base class method).
        
        Args:
            tree (Node): Tree to evaluate
            X (pandas.DataFrame): Time series features
            y (pandas.DataFrame): Reward calculation data
            fitness_function (function): Fitness function
            is_training (bool): Whether this is for training or prediction
            **fitness_function_kwargs: Additional arguments for fitness function
        
        Returns:
            Node: Tree with fitness calculated
        """
        # Clear previous evaluation
        tree.clear_evaluation()
        
        # Calculate fitness using our custom fitness function signature
        tree.fitness = fitness_function(tree, X, y, **fitness_function_kwargs)
        
        return tree

    def predict_actions(self, X):
        """
        Predict action sequence for time series data.

        Args:
            X (pandas.DataFrame): Time series features

        Returns:
            list: Sequence of selected actions (actual actions, not indices)
        """
        if self._tree is None:
            raise ValueError("Action selector must be fitted before making predictions")
        
        action_indices = []
        for t in range(len(X)):
            action_idx = self._tree.predict_one(X.iloc[t], train=False)
            # Ensure valid action index
            if action_idx < 0 or action_idx >= len(self.action_space):
                action_idx = 0
            action_indices.append(int(action_idx))
        
        # Convert indices to actual actions
        actions = [self.action_space[idx] for idx in action_indices]
        return actions

    def predict_action_indices(self, X):
        """
        Predict action indices for time series data.

        Args:
            X (pandas.DataFrame): Time series features

        Returns:
            list: Sequence of action indices
        """
        if self._tree is None:
            raise ValueError("Action selector must be fitted before making predictions")
        
        action_indices = []
        for t in range(len(X)):
            action_idx = self._tree.predict_one(X.iloc[t], train=False)
            # Ensure valid action index
            if action_idx < 0 or action_idx >= len(self.action_space):
                action_idx = 0
            action_indices.append(int(action_idx))
        
        return action_indices

    def simulate_rewards(self, X, Y, reward_function=None):
        """
        Simulate the action sequence and calculate total rewards.

        Args:
            X (pandas.DataFrame): Time series features
            Y (pandas.DataFrame): Reward calculation data
            reward_function (callable, optional): Custom reward function (uses default if None)

        Returns:
            tuple: (total_reward, individual_rewards, actions)
        """
        if reward_function is None:
            reward_function = self.reward_function
        
        actions = self.predict_actions(X)
        total_reward = 0.0
        individual_rewards = []
        
        for t, action in enumerate(actions):
            reward = reward_function(X.iloc[t], action, Y.iloc[t], t)
            discounted_reward = reward * (self.discount_factor ** t)
            individual_rewards.append(reward)
            total_reward += discounted_reward
        
        return total_reward, individual_rewards, actions

    def get_action_distribution(self, X):
        """
        Get the distribution of actions predicted for the time series.

        Args:
            X (pandas.DataFrame): Time series features

        Returns:
            dict: Dictionary mapping actions to their frequencies
        """
        actions = self.predict_actions(X)
        action_counts = {}
        
        for action in self.action_space:
            action_counts[action] = actions.count(action)
        
        return action_counts

    def plot_fitness_evolution(self):
        """
        Plot the evolution of fitness and rewards during training.
        Requires matplotlib.
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Plot fitness evolution
            ax1.plot(self._best_fitness, label='Best Fitness', color='blue')
            ax1.plot(self._avg_fitness, label='Average Fitness', color='red', alpha=0.7)
            ax1.set_xlabel('Generation')
            ax1.set_ylabel('Fitness (lower is better)')
            ax1.set_title('Fitness Evolution')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot reward evolution
            ax2.plot(self._best_rewards, label='Best Total Reward', color='green')
            ax2.set_xlabel('Generation')
            ax2.set_ylabel('Total Reward')
            ax2.set_title('Reward Evolution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not available - cannot plot fitness evolution")

    def __str__(self):
        """String representation of the action selector."""
        if self._tree is None:
            return f"GATreeActionSelector(actions={self.action_space}, unfitted)"
        
        try:
            # Add timeout protection for potentially problematic tree operations
            depth = self._tree.max_depth()
            size = self._tree.size()
            return (f"GATreeActionSelector(actions={self.action_space}, "
                    f"depth={depth}, size={size})")
        except Exception as e:
            return f"GATreeActionSelector(actions={self.action_space}, error={str(e)})"

    def __repr__(self):
        """Detailed representation of the action selector."""
        return self.__str__()