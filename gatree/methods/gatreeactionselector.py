import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator

# GATree
from gatree.tree.node import Node
from gatree.ga.selection import Selection
from gatree.ga.crossover import Crossover
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
        min_samples_leaf (int, optional): Minimum number of samples required to be at a leaf node (default: 1)
        discount_factor (float, optional): Future reward discount factor (default: 1.0)
        random (Random, optional): Random number generator
        n_jobs (int, optional): Number of jobs to run in parallel
        random_state (int, optional): Seed for reproducibility

    Attributes:
        action_space (list): List of possible actions
        reward_function (callable): Reward calculation function
        max_depth (int): Maximum depth of the tree
        min_samples_leaf (int): Minimum number of samples required to be at a leaf node
        discount_factor (float): Discount factor for future rewards
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

    def __init__(self, action_space, reward_function, max_depth=None, min_samples_leaf=1,
                 discount_factor=1.0, discount_direction='forward', global_reward_function=None, 
                 global_reward_weight=1.0, random=None, n_jobs=1, random_state=None):
        """
        Initialize the Genetic Algorithm Action Selector.

        Args:
            action_space (list): List of possible actions
            reward_function (callable): Function(state, action, y_data, timestep, previous_action) -> reward
            max_depth (int, optional): Maximum depth of the tree
            min_samples_leaf (int, optional): Minimum number of samples required to be at a leaf node (default: 1)
            discount_factor (float, optional): Future reward discount factor
            discount_direction (str, optional): 'forward' (standard) or 'backward' (reverse discount)
            global_reward_function (callable, optional): Function(rewards, actions, X, Y) -> global_reward
            global_reward_weight (float, optional): Weight for global reward component
            random (Random, optional): Random number generator
            n_jobs (int, optional): Number of jobs to run in parallel
            random_state (int, optional): Seed for reproducibility
        """
        super().__init__(max_depth, random, None, n_jobs, random_state)  # fitness_function set later
        
        self.action_space = action_space
        self.reward_function = reward_function
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.discount_factor = discount_factor
        self.discount_direction = discount_direction
        self.global_reward_function = global_reward_function
        self.global_reward_weight = global_reward_weight
        self.action_count = len(action_space)
        self._best_rewards = []
        
        # Validate parameters
        if discount_direction not in ['forward', 'backward']:
            raise ValueError("discount_direction must be 'forward' or 'backward'")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be at least 1")
        
        # Set the fitness function to our reward-based function
        self.fitness_function = self.default_fitness_function

    @staticmethod
    def default_fitness_function(root, X, Y, action_space, reward_function, discount_factor=1.0, 
                                discount_direction='forward', global_reward_function=None, 
                                global_reward_weight=1.0, sample_weight=None, **kwargs):
        """
        Default fitness function for action selection based on cumulative rewards.
        
        Simulates the action sequence determined by the tree and calculates
        the total discounted reward plus optional global reward minus complexity penalty.

        Args:
            root (Node): Root node of the tree
            X (pandas.DataFrame): Time series features
            Y (pandas.DataFrame): Reward calculation data
            action_space (list): List of possible actions
            reward_function (callable): Reward calculation function
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
        
        # Simulate action sequence and calculate rewards
        previous_action = None  # Initialize previous action as None for first timestep
        for t in range(n_timesteps):
            try:
                # Get action index from tree prediction
                action_idx = int(root.predict_one(X.iloc[t], train=False))
                
                # Ensure action index is valid
                if action_idx < 0 or action_idx >= len(action_space):
                    action_idx = 0  # Default to first action
                
                # Get actual action
                action = action_space[action_idx]
                actions.append(action)
                
                # Calculate reward for this timestep (now includes previous_action)
                reward = reward_function(X.iloc[t], action, Y.iloc[t], t, previous_action)
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
                    # Discount from the end: (n_timesteps - 1 - t)
                    reverse_t = n_timesteps - 1 - t
                    discounted_reward = reward * (discount_factor ** reverse_t)
                else:
                    # No discounting (fallback)
                    discounted_reward = reward
                
                total_reward += discounted_reward
                
                # Update previous_action for next iteration
                previous_action = action
                
            except Exception as e:
                # Handle any errors in reward calculation
                print(f"Error calculating reward at timestep {t}: {e}")
                # Use a smaller penalty to avoid dominating the fitness
                total_reward -= 1000  # Smaller penalty for errors
        
        # Calculate global reward if function provided
        global_reward = 0.0
        if global_reward_function is not None:
            try:
                global_reward = global_reward_function(individual_rewards, actions, X, Y) * global_reward_weight
                # Apply sample weights to global reward if provided
                if sample_weight is not None:
                    # Use mean weight for global reward
                    global_reward = global_reward * np.mean(sample_weight)
            except Exception as e:
                print(f"Error in global reward function: {e}")
                global_reward = -100  # Penalty for errors in global function
        
        # Add complexity penalty (encourage simpler trees)
        complexity_penalty = 0.001 * root.size()
        
        # Return negative reward (since GA minimizes fitness)
        # Formula: fitness = -(total_discounted_reward + global_reward) + complexity_penalty
        fitness = -(total_reward + global_reward) + complexity_penalty
        
        return fitness

    def fit(self, X, Y, sample_weight=None, population_size=100, max_iter=1000, mutation_probability=0.15, 
            elite_size=2, selection_tournament_size=3, fitness_function_kwargs={}, progress_callback=None,
            early_stopping=False, patience=50, min_delta=1e-4, restore_best_weights=True):
        """
        Fit the action selector to time series data.

        Args:
            X (pandas.DataFrame): Time series features (rows = timesteps, cols = features)
            Y (pandas.DataFrame): Additional data for reward calculation
            sample_weight (array-like, optional): Sample weights for timesteps. If None, all timesteps have equal weight.
            population_size (int, optional): Size of the population
            max_iter (int, optional): Maximum number of iterations
            mutation_probability (float, optional): Probability of mutation
            elite_size (int, optional): Number of elite trees
            selection_tournament_size (int, optional): Tournament size for selection
            fitness_function_kwargs (dict, optional): Additional kwargs for fitness function
            progress_callback (callable, optional): Function called after each generation with (generation, best_fitness, avg_fitness, best_reward).
            early_stopping (bool, optional): Whether to use early stopping (default: False).
            patience (int, optional): Number of generations to wait for improvement before stopping (default: 50).
            min_delta (float, optional): Minimum change in fitness to qualify as an improvement (default: 1e-4).
            restore_best_weights (bool, optional): Whether to restore the best tree when early stopping (default: True).

        Returns:
            self: The fitted action selector
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
                'discount_factor': self.discount_factor,
                'discount_direction': self.discount_direction,
                'global_reward_function': self.global_reward_function,
                'global_reward_weight': self.global_reward_weight
            })

            # Generation of initial population
            population = []
            for _ in range(population_size):
                try:
                    tree = self._make_node_with_min_samples(
                        max_depth=self.max_depth,
                        min_samples_leaf=self.min_samples_leaf,
                        random=self.random,
                        att_indexes=self.att_indexes, 
                        att_values=self.att_values, 
                        class_count=self.class_count,
                        n_samples=X.shape[0]
                    )
                    population.append(tree)
                except Exception as e:
                    # If tree generation fails, create a simple leaf node with random action
                    fallback_action = self.random.choice(list(range(self.action_count)))
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
                        delayed(GATreeActionSelector._predict_and_evaluate)(
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
                    # Fitness = -total_reward + complexity_penalty
                    # So: total_reward = -fitness + complexity_penalty
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
                            if progress_callback is not None:
                                try:
                                    progress_callback(i, best_fitness, avg_fitness, best_reward)
                                except Exception:
                                    print(f"Warning: Progress callback failed at generation {i}")
                            print(f"Early stopping at generation {i} (patience={patience})")
                            break
                    
                    # Call progress callback if provided
                    if progress_callback is not None:
                        try:
                            progress_callback(i, best_fitness, avg_fitness, best_reward)
                        except Exception:
                            print(f"Warning: Progress callback failed at generation {i}")

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
                                    mutation1 = self._mutate_with_min_samples(
                                        root=crossover1, 
                                        att_indexes=self.att_indexes,
                                        att_values=self.att_values, 
                                        class_count=self.class_count,
                                        random=self.random,
                                        max_depth=self.max_depth,
                                        min_samples_leaf=self.min_samples_leaf,
                                        n_samples=X.shape[0]
                                    )
                                
                                if self.random.random() < mutation_probability:
                                    mutation2 = self._mutate_with_min_samples(
                                        root=crossover2, 
                                        att_indexes=self.att_indexes,
                                        att_values=self.att_values, 
                                        class_count=self.class_count,
                                        random=self.random,
                                        max_depth=self.max_depth,
                                        min_samples_leaf=self.min_samples_leaf,
                                        n_samples=X.shape[0]
                                    )

                                # Add new trees to descendant population
                                descendant.extend([mutation1, mutation2])
                            except Exception:
                                # If generation fails, add copies of elite trees
                                print(f"Warning: Generation failed at iteration {i}, using elite copies")
                                if len(elites) > 0:
                                    descendant.extend([Node.copy(elites[0]), Node.copy(elites[0])])

                        # Elites + descendants
                        descendant.sort(key=lambda x: x.fitness, reverse=False)
                        descendant = elites + descendant[:population_size - elite_size]

                        # Replace old population with new population
                        population = descendant
                        
                except Exception:
                    print(f"Warning: Error in generation {i}")
                    # Continue with current population
                    continue

            # Set the final tree (best tree if early stopping with restore_best_weights, otherwise current best)
            if early_stopping and restore_best_weights and best_tree is not None:
                self._tree = best_tree
            else:
                self._tree = population[0]
            
            return self
            
        except Exception as e:
            raise RuntimeError(f"Failed to fit GATreeActionSelector: {str(e)}") from e

    def _make_node_with_min_samples(self, depth=1, max_depth=None, min_samples_leaf=1, 
                                   random=None, att_indexes=None, att_values=None, 
                                   class_count=None, n_samples=None, current_samples=None):
        """
        Generate a node that respects min_samples_leaf constraint.
        
        Args:
            depth (int): Current depth of the tree
            max_depth (int, optional): Maximum depth of the tree
            min_samples_leaf (int): Minimum samples required at leaf nodes
            random (Random): Random number generator
            att_indexes (numpy.ndarray): Attribute indexes
            att_values (dict): Attribute values
            class_count (int): Number of classes
            n_samples (int): Total number of samples in dataset
            current_samples (int, optional): Current number of samples at this node
            
        Returns:
            Node: Generated node with min_samples_leaf constraint
        """
        if current_samples is None:
            current_samples = n_samples
            
        if max_depth is None:
            max_depth = depth + 2

        # Force leaf if we're at max depth or don't have enough samples to split
        should_be_leaf = (depth >= max_depth or 
                         current_samples < 2 * min_samples_leaf or
                         len(att_indexes) == 0)
        
        # Also randomly decide to be leaf (but respect constraints)
        if not should_be_leaf and depth > 1:
            should_be_leaf = random.choice([True, False])

        if should_be_leaf:
            # Create leaf node
            r = Node._rand_index(random, class_count)
            return Node(att_index=-1, att_value=r)
        
        # Try to create internal node
        try:
            subset_index = Node._rand_index(random, len(att_indexes))
            att_index = att_indexes[subset_index]
            
            if att_index in att_values and len(att_values[att_index]) > 0:
                value_index = Node._rand_index(random, len(att_values[att_index]))
                att_value = att_values[att_index][value_index]
                node = Node(att_index=att_index, att_value=att_value)
                
                # Estimate samples for children (simplified approach)
                # In practice, this would require actual data splitting
                left_samples = max(min_samples_leaf, current_samples // 2)
                right_samples = max(min_samples_leaf, current_samples - left_samples)
                
                # Create children
                left_child = self._make_node_with_min_samples(
                    depth=depth + 1, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                    random=random, att_indexes=att_indexes, att_values=att_values,
                    class_count=class_count, n_samples=n_samples, current_samples=left_samples
                )
                right_child = self._make_node_with_min_samples(
                    depth=depth + 1, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                    random=random, att_indexes=att_indexes, att_values=att_values,
                    class_count=class_count, n_samples=n_samples, current_samples=right_samples
                )
                
                # Ensure children are not None
                if left_child is None:
                    left_child = Node(att_index=-1, att_value=Node._rand_index(random, class_count))
                if right_child is None:
                    right_child = Node(att_index=-1, att_value=Node._rand_index(random, class_count))
                    
                node.set_left(left_child)
                node.set_right(right_child)
                return node
            else:
                # No valid attribute values, create a leaf
                r = Node._rand_index(random, class_count)
                return Node(att_index=-1, att_value=r)
                
        except Exception as e:
            # Create a simple leaf node as fallback
            r = Node._rand_index(random, class_count)
            return Node(att_index=-1, att_value=r)

    def _mutate_with_min_samples(self, root, att_indexes, att_values, class_count, 
                                random, max_depth=None, min_samples_leaf=1, n_samples=None):
        """
        Custom mutation that respects min_samples_leaf constraint.
        
        Args:
            root (Node): The root node of the tree
            att_indexes (list): List of attribute indexes
            att_values (list): List of attribute values
            class_count (int): Number of classes
            random (Random): Random number generator
            max_depth (int, optional): Maximum depth constraint
            min_samples_leaf (int): Minimum samples per leaf constraint
            n_samples (int): Total number of samples in dataset
            
        Returns:
            Node: Mutated tree respecting min_samples_leaf
        """
        # Use standard mutation but replace any new subtree generation with our custom method
        from gatree.ga.mutation import Mutation
        
        # Create a copy of the tree
        node = Node.copy(root)
        depth = node.max_depth()

        # Find a node to mutate
        current = node
        while True:
            if current.att_index == -1:  # for leaves
                # For leaf mutations, we can safely use standard operations
                # since they don't create new subtrees that might violate constraints
                if random.choice([True, False]):
                    # Change class (safe operation)
                    self._change_class_safe(current, class_count, random)
                else:
                    # Exchange for new subtree (use our custom method)
                    self._exchange_class_for_tree_safe(current, att_indexes, att_values, 
                                                     class_count, random, max_depth, 
                                                     min_samples_leaf, n_samples)
                break
            elif random.randint(0, depth) == 0:  # for mid-tree nodes
                rand = random.random()
                if rand < 0.25 or current.parent is None:
                    # Change attribute (safe operation)
                    Mutation.change_attribute(current, att_indexes, att_values, random)
                elif rand < 0.5:
                    # Change attribute value (safe operation)
                    Mutation.change_attribute_value(current, att_values, random)
                elif rand < 0.75:
                    # Exchange for class (safe operation)
                    Mutation.exchange_tree_for_class(current, class_count, random)
                else:
                    # Exchange for new subtree (use our custom method)
                    self._exchange_tree_for_tree_safe(current, att_indexes, att_values,
                                                    class_count, random, max_depth,
                                                    min_samples_leaf, n_samples)
                break

            # Move to next random child
            if random.choice([True, False]):
                if current.left is not None:
                    current = current.left
                else:
                    break
            else:
                if current.right is not None:
                    current = current.right
                else:
                    break

        result = current.get_root()
        if max_depth is not None:
            result = Node.enforce_max_depth(result, max_depth, class_count, random)
        return result

    def _change_class_safe(self, node, class_count, random):
        """Safe class change that doesn't violate constraints."""
        result_old = node.att_value
        result_new = result_old
        while result_old == result_new:
            result_new = random.randint(0, class_count)
        node.att_value = result_new

    def _exchange_class_for_tree_safe(self, node, att_indexes, att_values, class_count, 
                                    random, max_depth, min_samples_leaf, n_samples):
        """Exchange leaf for subtree respecting min_samples_leaf."""
        parent = node.parent
        if parent is None:
            return
            
        left = (parent.left == node)
        
        # Use our custom tree generation method
        subtree = self._make_node_with_min_samples(
            depth=node.depth(), max_depth=max_depth, min_samples_leaf=min_samples_leaf,
            random=random, att_indexes=att_indexes, att_values=att_values,
            class_count=class_count, n_samples=n_samples, current_samples=min_samples_leaf
        )
        subtree.parent = parent

        if left:
            parent.set_left(subtree)
        else:
            parent.set_right(subtree)

    def _exchange_tree_for_tree_safe(self, node, att_indexes, att_values, class_count,
                                   random, max_depth, min_samples_leaf, n_samples):
        """Exchange subtree for new subtree respecting min_samples_leaf."""
        parent = node.parent
        if parent is None:
            return
            
        left = (parent.left == node)
        
        # Use our custom tree generation method
        subtree = self._make_node_with_min_samples(
            depth=node.depth(), max_depth=max_depth, min_samples_leaf=min_samples_leaf,
            random=random, att_indexes=att_indexes, att_values=att_values,
            class_count=class_count, n_samples=n_samples, current_samples=min_samples_leaf
        )
        subtree.parent = parent

        if left:
            parent.set_left(subtree)
        else:
            parent.set_right(subtree)

    @staticmethod
    def _predict_and_evaluate(tree, X, y, fitness_function, is_training=False, sample_weight=None, **fitness_function_kwargs):
        """
        Evaluate a tree on time series data (overrides base class method).
        
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
        
        previous_action = None  # Initialize previous action as None for first timestep
        for t, action in enumerate(actions):
            reward = reward_function(X.iloc[t], action, Y.iloc[t], t, previous_action)
            discounted_reward = reward * (self.discount_factor ** t)
            individual_rewards.append(reward)
            total_reward += discounted_reward
            previous_action = action  # Update previous action for next iteration
        
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

    def predict_action(self, X_instance):
        """
        Predict a single action for a given instance.
        
        Args:
            X_instance: Single instance (pandas Series or similar)
            
        Returns:
            int: Predicted action index
        """
        if self._tree is None:
            raise ValueError("Model must be fitted before making predictions")
        
        action_idx = self._tree.predict_one(X_instance, train=False)
        
        # Ensure action index is valid
        if action_idx < 0 or action_idx >= len(self.action_space):
            action_idx = 0  # Default to first action
            
        return action_idx

    def __str__(self):
        """String representation of the action selector."""
        if self._tree is None:
            discount_info = f"discount={self.discount_factor}"
            if self.discount_direction == 'backward':
                discount_info += " (backward)"
            global_info = ""
            if self.global_reward_function is not None:
                global_info = f", global_reward=True"
            return f"GATreeActionSelector(actions={self.action_space}, {discount_info}{global_info}, min_samples_leaf={self.min_samples_leaf}, unfitted)"
        
        try:
            # Add timeout protection for potentially problematic tree operations
            depth = self._tree.max_depth()
            size = self._tree.size()
            discount_info = f"discount={self.discount_factor}"
            if self.discount_direction == 'backward':
                discount_info += " (backward)"
            global_info = ""
            if self.global_reward_function is not None:
                global_info = f", global_reward=True"
            return (f"GATreeActionSelector(actions={self.action_space}, "
                    f"{discount_info}{global_info}, min_samples_leaf={self.min_samples_leaf}, depth={depth}, size={size})")
        except Exception as e:
            return f"GATreeActionSelector(actions={self.action_space}, min_samples_leaf={self.min_samples_leaf}, error={str(e)})"

    def __repr__(self):
        """Detailed representation of the action selector."""
        return self.__str__()

    def predict_actions_with_path(self, X):
        """
        Predict action sequence and return decision paths for interpretability.
        
        Args:
            X (pandas.DataFrame): Time series features
            
        Returns:
            tuple: (actions, paths) where actions are the actual actions and paths contain traversal info
        """
        if self._tree is None:
            raise ValueError("Action selector must be fitted before making predictions")
        
        actions = []
        paths = []
        
        for t in range(len(X)):
            action_idx, path = self._tree.predict_one_with_path(X.iloc[t], train=False)
            # Ensure valid action index
            if action_idx < 0 or action_idx >= len(self.action_space):
                action_idx = 0
            actions.append(self.action_space[action_idx])
            paths.append(path)
        
        return actions, paths

    def predict_single_action_with_path(self, X_instance):
        """
        Predict a single action and return the decision path.
        
        Args:
            X_instance: Single instance (pandas Series or similar)
            
        Returns:
            tuple: (action, path) where action is the predicted action and path contains decision steps
        """
        if self._tree is None:
            raise ValueError("Action selector must be fitted before making predictions")
        
        action_idx, path = self._tree.predict_one_with_path(X_instance, train=False)
        
        # Ensure valid action index
        if action_idx < 0 or action_idx >= len(self.action_space):
            action_idx = 0
        
        action = self.action_space[action_idx]
        return action, path
