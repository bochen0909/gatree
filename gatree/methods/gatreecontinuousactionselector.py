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


class GATreeContinuousActionSelector(GATree, BaseEstimator):
    """
    Evolutionary decision tree for continuous action selection in time series.
    
    This class evolves decision trees that can make optimal continuous actions (bounded between 0 and 1)
    at each time step to maximize cumulative rewards over a time series. Unlike discrete action selection,
    the leaf nodes output continuous values that are interpreted as actions.

    Args:
        reward_function (callable): Function to calculate rewards
        action_bounds (tuple, optional): (min, max) bounds for actions (default: (0, 1))
        max_depth (int, optional): Maximum depth of the tree
        discount_factor (float, optional): Future reward discount factor (default: 1.0)
        discount_direction (str, optional): 'forward' or 'backward' discounting
        global_reward_function (callable, optional): Global reward function
        global_reward_weight (float, optional): Weight for global reward component
        n_action_bins (int, optional): Number of discrete bins for continuous actions (default: 50)
        random (Random, optional): Random number generator
        n_jobs (int, optional): Number of jobs to run in parallel
        random_state (int, optional): Seed for reproducibility

    Attributes:
        reward_function (callable): Reward calculation function
        action_bounds (tuple): (min, max) bounds for actions
        discount_factor (float): Discount factor for future rewards
        discount_direction (str): Direction of discounting
        global_reward_function (callable): Global reward function
        global_reward_weight (float): Weight for global reward
        n_action_bins (int): Number of discrete action bins
        max_depth (int): Maximum depth of the tree
        random (Random): Random number generator
        X (pandas.DataFrame): Training time series features
        Y (pandas.DataFrame): Training reward calculation data
        att_indexes (numpy.ndarray): Array of attribute indexes
        att_values (dict): Dictionary of attribute values
        n_jobs (int): Number of jobs to run in parallel
        random_state (int): Seed for reproducibility
        _tree (Node): The fitted tree
        _best_fitness (list): List of best fitness values for each iteration
        _avg_fitness (list): List of average fitness values for each iteration
        _best_rewards (list): List of best total rewards for each iteration
    """

    def __init__(self, reward_function, action_bounds=(0, 1), max_depth=None, discount_factor=1.0, 
                 discount_direction='forward', global_reward_function=None, global_reward_weight=1.0,
                 n_action_bins=50, random=None, n_jobs=1, random_state=None):
        """
        Initialize the Genetic Algorithm Continuous Action Selector.

        Args:
            reward_function (callable): Function(state, action, y_data, timestep, previous_action) -> reward
            action_bounds (tuple, optional): (min, max) bounds for continuous actions
            max_depth (int, optional): Maximum depth of the tree
            discount_factor (float, optional): Future reward discount factor
            discount_direction (str, optional): 'forward' (standard) or 'backward' (reverse discount)
            global_reward_function (callable, optional): Function(rewards, actions, X, Y) -> global_reward
            global_reward_weight (float, optional): Weight for global reward component
            n_action_bins (int, optional): Number of discrete bins for continuous actions
            random (Random, optional): Random number generator
            n_jobs (int, optional): Number of jobs to run in parallel
            random_state (int, optional): Seed for reproducibility
        """
        super().__init__(max_depth, random, None, n_jobs, random_state)  # fitness_function set later
        
        self.reward_function = reward_function
        self.action_bounds = action_bounds
        self.discount_factor = discount_factor
        self.discount_direction = discount_direction
        self.global_reward_function = global_reward_function
        self.global_reward_weight = global_reward_weight
        self.n_action_bins = n_action_bins
        self._best_rewards = []
        
        # Validate action bounds
        if len(action_bounds) != 2 or action_bounds[0] >= action_bounds[1]:
            raise ValueError("action_bounds must be (min, max) with min < max")
        
        # Validate discount direction
        if discount_direction not in ['forward', 'backward']:
            raise ValueError("discount_direction must be 'forward' or 'backward'")
        
        # Validate n_action_bins
        if n_action_bins <= 0:
            raise ValueError("n_action_bins must be positive")
        
        # Set the fitness function to our reward-based function
        self.fitness_function = self.default_fitness_function

    def _action_index_to_continuous(self, action_index):
        """
        Convert discrete action index to continuous action value.
        
        Args:
            action_index (int): Discrete action index
            
        Returns:
            float: Continuous action value within bounds
        """
        # Ensure action index is valid
        action_index = max(0, min(action_index, self.n_action_bins - 1))
        
        # Map index to continuous value
        min_val, max_val = self.action_bounds
        continuous_action = min_val + (action_index / (self.n_action_bins - 1)) * (max_val - min_val)
        
        return continuous_action

    def _continuous_to_action_index(self, continuous_action):
        """
        Convert continuous action value to discrete action index.
        
        Args:
            continuous_action (float): Continuous action value
            
        Returns:
            int: Discrete action index
        """
        # Clip to bounds
        min_val, max_val = self.action_bounds
        continuous_action = max(min_val, min(continuous_action, max_val))
        
        # Map to index
        normalized = (continuous_action - min_val) / (max_val - min_val)
        action_index = int(normalized * (self.n_action_bins - 1))
        
        return action_index

    @staticmethod
    def default_fitness_function(root, X, Y, reward_function, action_bounds, n_action_bins, 
                                discount_factor=1.0, discount_direction='forward', 
                                global_reward_function=None, global_reward_weight=1.0, 
                                sample_weight=None, **kwargs):
        """
        Default fitness function for continuous action selection based on cumulative rewards.
        
        Simulates the continuous action sequence determined by the tree and calculates
        the total discounted reward plus optional global reward minus complexity penalty.

        Args:
            root (Node): Root node of the tree
            X (pandas.DataFrame): Time series features
            Y (pandas.DataFrame): Reward calculation data
            reward_function (callable): Reward calculation function
            action_bounds (tuple): (min, max) bounds for actions
            n_action_bins (int): Number of discrete action bins
            discount_factor (float): Discount factor for rewards
            discount_direction (str): 'forward' (standard) or 'backward' (reverse discount)
            global_reward_function (callable, optional): Global reward function
            global_reward_weight (float): Weight for global reward component
            sample_weight (array-like, optional): Sample weights for timesteps

        Returns:
            float: Fitness value (lower is better, so we negate rewards)
        """
        if root is None or not callable(getattr(root, "predict_one", None)):
            return float('inf')
        if getattr(root, "att_index", None) is None and getattr(root, "att_value", None) is None:
            return float('inf')

        total_reward = 0.0
        n_timesteps = len(X)
        individual_rewards = []
        actions = []
        
        # Validate sample weights if provided
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)
            if len(sample_weight) != n_timesteps:
                raise ValueError(f"sample_weight length {len(sample_weight)} != n_timesteps {n_timesteps}")
        
        # Helper function to convert action index to continuous value
        def action_index_to_continuous(action_index):
            action_index = max(0, min(action_index, n_action_bins - 1))
            min_val, max_val = action_bounds
            return min_val + (action_index / (n_action_bins - 1)) * (max_val - min_val)
        
        # Simulate action sequence and calculate rewards
        previous_action = None  # Initialize previous action as None for first timestep
        for t in range(n_timesteps):
            try:
                # Get action index from tree prediction
                action_idx = int(root.predict_one(X.iloc[t], train=False))
                
                # Convert to continuous action
                continuous_action = action_index_to_continuous(action_idx)
                actions.append(continuous_action)
                
                # Calculate reward for this timestep
                reward = reward_function(X.iloc[t], continuous_action, Y.iloc[t], t, previous_action)
                individual_rewards.append(reward)
                
                # Apply sample weight if provided
                if sample_weight is not None:
                    reward = reward * sample_weight[t]
                
                # Apply discount factor based on direction
                if discount_direction == 'forward':
                    # Standard discounting: earlier rewards worth more
                    discounted_reward = reward * (discount_factor ** t)
                elif discount_direction == 'backward':
                    # Reverse discounting: later rewards worth more
                    reverse_t = n_timesteps - 1 - t
                    discounted_reward = reward * (discount_factor ** reverse_t)
                else:
                    # No discounting (fallback)
                    discounted_reward = reward
                
                total_reward += discounted_reward
                
                # Update previous_action for next iteration
                previous_action = continuous_action
                
            except Exception as e:
                # Handle any errors in reward calculation
                print(f"Error calculating reward at timestep {t}: {e}")
                total_reward -= 1000  # Penalty for errors
        
        # Calculate global reward if function provided
        global_reward = 0.0
        if global_reward_function is not None:
            try:
                global_reward = global_reward_function(individual_rewards, actions, X, Y) * global_reward_weight
                # Apply sample weights to global reward if provided
                if sample_weight is not None:
                    global_reward = global_reward * np.mean(sample_weight)
            except Exception as e:
                print(f"Error in global reward function: {e}")
                global_reward = -100  # Penalty for errors in global function
        
        # Add complexity penalty (encourage simpler trees)
        complexity_penalty = 0.001 * root.size()
        
        # Return negative reward (since GA minimizes fitness)
        fitness = -(total_reward + global_reward) + complexity_penalty
        
        return fitness

    def fit(self, X, Y, sample_weight=None, population_size=100, max_iter=1000, mutation_probability=0.15, 
            elite_size=2, selection_tournament_size=3, fitness_function_kwargs={}, progress_callback=None,
            early_stopping=False, patience=50, min_delta=1e-4, restore_best_weights=True):
        """
        Fit the continuous action selector to time series data.

        Args:
            X (pandas.DataFrame): Time series features (rows = timesteps, cols = features)
            Y (pandas.DataFrame): Additional data for reward calculation
            sample_weight (array-like, optional): Sample weights for timesteps
            population_size (int, optional): Size of the population
            max_iter (int, optional): Maximum number of iterations
            mutation_probability (float, optional): Probability of mutation
            elite_size (int, optional): Number of elite trees
            selection_tournament_size (int, optional): Tournament size for selection
            fitness_function_kwargs (dict, optional): Additional kwargs for fitness function
            progress_callback (callable, optional): Function called after each generation
            early_stopping (bool, optional): Whether to use early stopping
            patience (int, optional): Number of generations to wait for improvement
            min_delta (float, optional): Minimum change in fitness to qualify as improvement
            restore_best_weights (bool, optional): Whether to restore best tree when early stopping

        Returns:
            self: The fitted continuous action selector
        """
        try:
            self.X = X
            self.Y = Y
            self.sample_weight = sample_weight
            self.att_indexes = np.arange(X.shape[1])
            
            # Validate sample weights
            if sample_weight is not None:
                sample_weight = np.asarray(sample_weight)
                if sample_weight.shape[0] != X.shape[0]:
                    raise ValueError(f"sample_weight must have same length as X: {sample_weight.shape[0]} != {X.shape[0]}")
                if np.any(sample_weight < 0):
                    raise ValueError("sample_weight must be non-negative")
                if np.sum(sample_weight) == 0:
                    raise ValueError("sample_weight cannot sum to zero")
            
            # Validate input parameters
            if population_size <= 0:
                raise ValueError("population_size must be positive")
            if max_iter < 0:
                raise ValueError("max_iter must be non-negative")
            if not 0 <= mutation_probability <= 1:
                raise ValueError("mutation_probability must be between 0 and 1")
            if elite_size < 0 or elite_size >= population_size:
                raise ValueError("elite_size must be between 0 and population_size")
            if selection_tournament_size <= 0:
                raise ValueError("selection_tournament_size must be positive")
            if early_stopping:
                if patience <= 0:
                    raise ValueError("patience must be positive when early_stopping is True")
                if min_delta < 0:
                    raise ValueError("min_delta must be non-negative")
            
            # Early stopping variables
            best_fitness_so_far = float('inf')
            best_tree = None
            patience_counter = 0
            stopped_early = False
            
            # Create split thresholds for features
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
            
            # Action indices for leaf nodes (will be converted to continuous values)
            self.att_values[-1] = list(range(self.n_action_bins))
            self.class_count = self.n_action_bins
            
            # Add required parameters to fitness function kwargs
            fitness_function_kwargs.update({
                'reward_function': self.reward_function,
                'action_bounds': self.action_bounds,
                'n_action_bins': self.n_action_bins,
                'discount_factor': self.discount_factor,
                'discount_direction': self.discount_direction,
                'global_reward_function': self.global_reward_function,
                'global_reward_weight': self.global_reward_weight
            })

            # Generation of initial population
            node = Node()
            population = []
            for _ in range(population_size):
                try:
                    tree = node.make_node(
                        max_depth=self.max_depth, 
                        random=self.random,
                        att_indexes=self.att_indexes, 
                        att_values=self.att_values, 
                        class_count=self.class_count
                    )
                    population.append(tree)
                except Exception as e:
                    # If tree generation fails, create a simple leaf node with random action
                    fallback_action = self.random.choice(list(range(self.n_action_bins)))
                    fallback_tree = Node(att_index=-1, att_value=fallback_action)
                    population.append(fallback_tree)

            # Evolution loop
            for i in range(max_iter + 1):
                try:
                    # Clear previous evaluation
                    for tree in population:
                        tree.clear_evaluation()

                    # Evaluation of population
                    population = Parallel(n_jobs=self.n_jobs)(
                        delayed(GATreeContinuousActionSelector._predict_and_evaluate)(
                            tree, X, Y, self.fitness_function, True, sample_weight, **fitness_function_kwargs
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
                    if best_fitness != float('inf'):
                        complexity_penalty = 0.001 * population[0].size()
                        best_reward = -best_fitness + complexity_penalty
                    else:
                        best_reward = float('-inf')  # Invalid tree
                    self._best_rewards.append(best_reward)
                    
                    # Early stopping check
                    if early_stopping:
                        if best_fitness < best_fitness_so_far - min_delta:
                            best_fitness_so_far = best_fitness
                            if restore_best_weights:
                                best_tree = Node.copy(population[0])
                            patience_counter = 0
                        else:
                            patience_counter += 1
                            
                        if patience_counter >= patience:
                            stopped_early = True
                            if progress_callback is not None:
                                try:
                                    progress_callback(i, best_fitness, avg_fitness, best_reward)
                                except Exception as callback_error:
                                    print(f"Warning: Progress callback failed at generation {i}: {callback_error}")
                            print(f"Early stopping at generation {i} (patience={patience})")
                            break
                    
                    # Call progress callback if provided
                    if progress_callback is not None:
                        try:
                            progress_callback(i, best_fitness, avg_fitness, best_reward)
                        except Exception as callback_error:
                            print(f"Warning: Progress callback failed at generation {i}: {callback_error}")

                    if i != max_iter:
                        # Elites
                        elites = population[:elite_size]

                        # Descendant generation
                        descendant = []
                        for _ in range(0, len(population), 2):
                            try:
                                # Tournament selection
                                tree1, tree2 = Selection.selection(
                                    population=population, 
                                    selection_tournament_size=selection_tournament_size, 
                                    random=self.random
                                )

                                # Crossover between selected trees
                                crossover1 = Crossover.crossover(
                                    tree1=tree1,
                                    tree2=tree2,
                                    random=self.random,
                                    max_depth=self.max_depth,
                                    class_count=self.class_count
                                )
                                crossover2 = Crossover.crossover(
                                    tree1=tree2,
                                    tree2=tree1,
                                    random=self.random,
                                    max_depth=self.max_depth,
                                    class_count=self.class_count
                                )

                                # Mutation of new trees
                                mutation1 = crossover1
                                mutation2 = crossover2
                                
                                if self.random.random() < mutation_probability:
                                    mutation1 = Mutation.mutation(
                                        root=crossover1, 
                                        att_indexes=self.att_indexes,
                                        att_values=self.att_values, 
                                        class_count=self.class_count,
                                        random=self.random,
                                        max_depth=self.max_depth
                                    )
                                
                                if self.random.random() < mutation_probability:
                                    mutation2 = Mutation.mutation(
                                        root=crossover2, 
                                        att_indexes=self.att_indexes,
                                        att_values=self.att_values, 
                                        class_count=self.class_count,
                                        random=self.random,
                                        max_depth=self.max_depth
                                    )

                                # Add new trees to descendant population
                                descendant.extend([mutation1, mutation2])
                            except Exception as generation_error:
                                # If generation fails, add copies of elite trees
                                print(f"Warning: Generation failed at iteration {i}, using elite copies: {generation_error}")
                                if len(elites) > 0:
                                    descendant.extend([Node.copy(elites[0]), Node.copy(elites[0])])

                        # Elites + descendants
                        descendant.sort(key=lambda x: x.fitness, reverse=False)
                        descendant = elites + descendant[:population_size - elite_size]

                        # Replace old population with new population
                        population = descendant
                        
                except Exception as iteration_error:
                    print(f"Warning: Error in generation {i}: {iteration_error}")
                    # Continue with current population
                    continue

            # Set the final tree
            if early_stopping and restore_best_weights and best_tree is not None:
                self._tree = best_tree
            else:
                self._tree = population[0]
            
            return self
            
        except Exception as e:
            raise RuntimeError(f"Failed to fit GATreeContinuousActionSelector: {str(e)}") from e

    @staticmethod
    def _predict_and_evaluate(tree, X, y, fitness_function, is_training=False, sample_weight=None, **fitness_function_kwargs):
        """
        Evaluate a tree on time series data.
        
        Args:
            tree (Node): Tree to evaluate
            X (pandas.DataFrame): Time series features
            y (pandas.DataFrame): Reward calculation data
            fitness_function (function): Fitness function
            is_training (bool): Whether this is for training or prediction
            sample_weight (array-like, optional): Sample weights
            **fitness_function_kwargs: Additional arguments for fitness function
        
        Returns:
            Node: Tree with fitness calculated
        """
        # Clear previous evaluation
        tree.clear_evaluation()
        
        # Pass sample weights to fitness function
        if sample_weight is not None:
            fitness_function_kwargs['sample_weight'] = sample_weight
        
        # Calculate fitness using our custom fitness function signature
        tree.fitness = fitness_function(tree, X, y, **fitness_function_kwargs)
        
        return tree

    def predict_actions(self, X):
        """
        Predict continuous action sequence for time series data.

        Args:
            X (pandas.DataFrame): Time series features

        Returns:
            list: Sequence of continuous actions within bounds
        """
        if self._tree is None:
            raise ValueError("Continuous action selector must be fitted before making predictions")
        
        continuous_actions = []
        for t in range(len(X)):
            action_idx = self._tree.predict_one(X.iloc[t], train=False)
            continuous_action = self._action_index_to_continuous(action_idx)
            continuous_actions.append(continuous_action)
        
        return continuous_actions

    def predict_action_indices(self, X):
        """
        Predict action indices for time series data (for internal use).

        Args:
            X (pandas.DataFrame): Time series features

        Returns:
            list: Sequence of action indices
        """
        if self._tree is None:
            raise ValueError("Continuous action selector must be fitted before making predictions")
        
        action_indices = []
        for t in range(len(X)):
            action_idx = self._tree.predict_one(X.iloc[t], train=False)
            # Ensure valid action index
            if action_idx < 0 or action_idx >= self.n_action_bins:
                action_idx = 0
            action_indices.append(int(action_idx))
        
        return action_indices

    def simulate_rewards(self, X, Y, reward_function=None):
        """
        Simulate the continuous action sequence and calculate total rewards.

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
        
        previous_action = None
        for t, action in enumerate(actions):
            reward = reward_function(X.iloc[t], action, Y.iloc[t], t, previous_action)
            
            # Apply discount factor based on direction
            if self.discount_direction == 'forward':
                discounted_reward = reward * (self.discount_factor ** t)
            elif self.discount_direction == 'backward':
                reverse_t = len(actions) - 1 - t
                discounted_reward = reward * (self.discount_factor ** reverse_t)
            else:
                discounted_reward = reward
            
            individual_rewards.append(reward)
            total_reward += discounted_reward
            previous_action = action
        
        return total_reward, individual_rewards, actions

    def get_action_statistics(self, X):
        """
        Get statistics of continuous actions predicted for the time series.

        Args:
            X (pandas.DataFrame): Time series features

        Returns:
            dict: Dictionary with action statistics (mean, std, min, max, etc.)
        """
        actions = self.predict_actions(X)
        actions_array = np.array(actions)
        
        return {
            'mean': np.mean(actions_array),
            'std': np.std(actions_array),
            'min': np.min(actions_array),
            'max': np.max(actions_array),
            'median': np.median(actions_array),
            'q25': np.percentile(actions_array, 25),
            'q75': np.percentile(actions_array, 75),
            'count': len(actions_array)
        }

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

    def plot_action_sequence(self, X, Y=None):
        """
        Plot the predicted continuous action sequence.
        Requires matplotlib.
        
        Args:
            X (pandas.DataFrame): Time series features
            Y (pandas.DataFrame, optional): Additional data for context
        """
        try:
            import matplotlib.pyplot as plt
            
            actions = self.predict_actions(X)
            
            plt.figure(figsize=(12, 6))
            plt.plot(actions, label='Predicted Actions', color='blue', linewidth=2)
            plt.axhline(y=self.action_bounds[0], color='red', linestyle='--', alpha=0.7, label=f'Lower Bound ({self.action_bounds[0]})')
            plt.axhline(y=self.action_bounds[1], color='red', linestyle='--', alpha=0.7, label=f'Upper Bound ({self.action_bounds[1]})')
            plt.xlabel('Timestep')
            plt.ylabel('Action Value')
            plt.title('Continuous Action Sequence')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not available - cannot plot action sequence")

    def predict_action(self, X_instance):
        """
        Predict a single continuous action for a given instance.
        
        Args:
            X_instance: Single instance (pandas Series or similar)
            
        Returns:
            float: Predicted continuous action value
        """
        if self._tree is None:
            raise ValueError("Model must be fitted before making predictions")
        
        action_idx = self._tree.predict_one(X_instance, train=False)
        continuous_action = self._action_index_to_continuous(action_idx)
        
        return continuous_action

    def __str__(self):
        """String representation of the continuous action selector."""
        if self._tree is None:
            discount_info = f"discount={self.discount_factor}"
            if self.discount_direction == 'backward':
                discount_info += " (backward)"
            global_info = ""
            if self.global_reward_function is not None:
                global_info = f", global_reward=True"
            return (f"GATreeContinuousActionSelector(bounds={self.action_bounds}, "
                    f"bins={self.n_action_bins}, {discount_info}{global_info}, unfitted)")
        
        try:
            depth = self._tree.max_depth()
            size = self._tree.size()
            discount_info = f"discount={self.discount_factor}"
            if self.discount_direction == 'backward':
                discount_info += " (backward)"
            global_info = ""
            if self.global_reward_function is not None:
                global_info = f", global_reward=True"
            return (f"GATreeContinuousActionSelector(bounds={self.action_bounds}, "
                    f"bins={self.n_action_bins}, {discount_info}{global_info}, "
                    f"depth={depth}, size={size})")
        except Exception as e:
            return f"GATreeContinuousActionSelector(bounds={self.action_bounds}, error={str(e)})"

    def __repr__(self):
        """Detailed representation of the continuous action selector."""
        return self.__str__()

    def predict_actions_with_path(self, X):
        """
        Predict continuous action sequence and return decision paths for interpretability.
        
        Args:
            X (pandas.DataFrame): Time series features
            
        Returns:
            tuple: (continuous_actions, paths) where paths contain traversal info
        """
        if self._tree is None:
            raise ValueError("Continuous action selector must be fitted before making predictions")
        
        continuous_actions = []
        paths = []
        
        for t in range(len(X)):
            action_idx, path = self._tree.predict_one_with_path(X.iloc[t], train=False)
            continuous_action = self._action_index_to_continuous(action_idx)
            continuous_actions.append(continuous_action)
            paths.append(path)
        
        return continuous_actions, paths

    def predict_single_action_with_path(self, X_instance):
        """
        Predict a single continuous action and return the decision path.
        
        Args:
            X_instance: Single instance (pandas Series or similar)
            
        Returns:
            tuple: (continuous_action, path) where path contains decision steps
        """
        if self._tree is None:
            raise ValueError("Continuous action selector must be fitted before making predictions")
        
        action_idx, path = self._tree.predict_one_with_path(X_instance, train=False)
        continuous_action = self._action_index_to_continuous(action_idx)
        
        return continuous_action, path