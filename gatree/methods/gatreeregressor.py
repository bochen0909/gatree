import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error
from sklearn.base import RegressorMixin

# GATree
from gatree.tree.node import Node
from gatree.ga.selection import Selection
from gatree.ga.crossover import Crossover
from gatree.ga.mutation import Mutation
from gatree.gatree import GATree


class GATreeRegressor(RegressorMixin, GATree):
    """
    Evolutionary decision tree regressor. The GATree regressor is a decision tree regressor that is trained using a genetic algorithm. The genetic algorithm is used to evolve a population of trees over multiple generations. The fitness of each tree is evaluated using a fitness function, which is used to select the best trees for crossover and mutation.

    Args:
        max_depth (int, optional): Maximum depth of the tree.
        random (Random, optional): Random number generator.
        fitness_function (function, optional): Fitness function for the genetic algorithm.
        n_jobs (int, optional): Number of jobs to run in parallel.
        random_state (int, optional): Seed for reproducibility.

    Attributes:
        max_depth (int, optional): Maximum depth of the tree.
        random (Random): Random number generator.
        X (pandas.DataFrame): Training data.
        y (pandas.Series): Target values.
        att_indexes (numpy.ndarray): Array of attribute indexes.
        att_values (dict): Dictionary of attribute values.
        y_min (float): Minimum target value.
        y_max (float): Maximum target value.
        fitness_function (function): Fitness function for the genetic algorithm.
        n_jobs (int): Number of jobs to run in parallel.
        random_state (int): Seed for reproducibility.
        _tree (Node): The fitted tree.
        _best_fitness (list): List of best fitness values for each iteration.
        _avg_fitness (list): List of average fitness values for each iteration.
    """

    def __init__(self, max_depth=None, random=None, fitness_function=None, n_jobs=1, random_state=None):
        """
        Initialise the Genetic Algorithm Tree Regressor. Maximum depth, random number generator, fitness function, number of jobs, and random state can be specified.

        Args:
            max_depth (int, optional): Maximum depth of the tree.
            random (Random, optional): Random number generator.
            fitness_function (function, optional): Fitness function for the genetic algorithm.
            random_state (int, optional): Seed reproducibility.
        """
        super().__init__(max_depth, random, fitness_function, n_jobs, random_state)

    @staticmethod
    def default_fitness_function(root, **fitness_function_kwargs):
        """
        Default fitness function for the genetic algorithm. Uses normalized mean squared error with complexity penalty.

        Args:
            root (Node): Root node of the tree.
            **fitness_function_kwargs: Additional arguments including sample_weight.

        Returns:
            float: The fitness value (lower is better).
        """
        if len(root.y_true) == 0 or len(root.y_pred) == 0:
            return float('inf')
        
        # Get sample weights if provided
        sample_weight = fitness_function_kwargs.get('sample_weight', None)
        
        # Calculate weighted MSE
        mse = mean_squared_error(root.y_true, root.y_pred, sample_weight=sample_weight)
        y_range = fitness_function_kwargs.get('y_range', 1.0)
        normalized_mse = mse / (y_range ** 2) if y_range > 0 else mse
        
        # Add complexity penalty
        complexity_penalty = 0.002 * root.size()
        
        return normalized_mse + complexity_penalty

    def fit(self, X, y, sample_weight=None, population_size=150, max_iter=2000, mutation_probability=0.1, elite_size=1,
            selection_tournament_size=2, fitness_function_kwargs={}, progress_callback=None):
        """
        Fit a tree to a training set. The population size, maximum iterations, mutation probability, elite size, and selection tournament size can be specified.

        Args:
            X (pandas.DataFrame): Training data.
            y (pandas.Series): Target values.
            sample_weight (array-like, optional): Sample weights. If None, all samples have equal weight.
            population_size (int, optional): Size of the population.
            max_iter (int, optional): Maximum number of iterations.
            mutation_probability (float, optional): Probability of mutation.
            elite_size (int, optional): Number of elite trees.
            selection_tournament_size (int, optional): Number of trees in tournament.
            fitness_function_kwargs (dict, optional): Additional kwargs to be passed to the fitness_function.
            progress_callback (callable, optional): Function called after each generation with (generation, best_fitness, avg_fitness).

        Returns:
            self: The fitted regressor.
        """
        try:
            self.X = X
            self.y = y
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
            
            # Create split thresholds for features (same as classification)
            self.att_values = {i: [(min_val + max_val) / 2 for min_val, max_val in zip(sorted(
                X.iloc[:, i].unique())[:-1], sorted(X.iloc[:, i].unique())[1:])] for i in range(X.shape[1])}
            
            # For regression, leaf values are continuous target values
            # Create a range of possible leaf values based on target distribution
            self.y_min = float(y.min())
            self.y_max = float(y.max())
            y_range = self.y_max - self.y_min
            
            # Create discrete leaf values by dividing target range into bins
            n_bins = min(50, len(y.unique()))  # Limit number of bins
            if n_bins > 1:
                leaf_values = np.linspace(self.y_min, self.y_max, n_bins).tolist()
            else:
                # For constant targets, create a small range around the value to avoid infinite loops
                leaf_values = [self.y_min - 0.1, self.y_min, self.y_min + 0.1]
            
            self.att_values[-1] = leaf_values
            self.class_count = len(leaf_values)
            
            # Add y_range to fitness function kwargs for normalization
            fitness_function_kwargs['y_range'] = y_range

            # Generation of initial population
            node = Node()
            population = []
            for _ in range(population_size):
                try:
                    new_tree = node.make_node(max_depth=self.max_depth, random=self.random,
                                  att_indexes=self.att_indexes, att_values=self.att_values, class_count=self.class_count)
                    population.append(new_tree)
                except Exception as e:
                    # If tree generation fails, create a simple leaf node
                    fallback_value = self.random.choice(leaf_values)
                    fallback_tree = Node(att_index=-1, att_value=fallback_value)
                    population.append(fallback_tree)

            for i in range(max_iter+1):
                try:
                    # Clear previous evaluation
                    for tree in population:
                        tree.clear_evaluation()

                    # Evaluation of population
                    population = Parallel(n_jobs=self.n_jobs)(delayed(GATreeRegressor._predict_and_evaluate)(
                        tree, X, y, self.fitness_function, True, sample_weight, **fitness_function_kwargs) for tree in population)

                    # Sort population by fitness (lower is better for regression)
                    population.sort(key=lambda x: x.fitness, reverse=False)

                    # Log best and average fitness
                    best_fitness = population[0].fitness
                    avg_fitness = sum([tree.fitness for tree in population]) / len(population)
                    
                    self._best_fitness.append(best_fitness)
                    self._avg_fitness.append(avg_fitness)
                    
                    # Call progress callback if provided
                    if progress_callback is not None:
                        try:
                            progress_callback(i, best_fitness, avg_fitness)
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
                                    population=population, selection_tournament_size=selection_tournament_size, random=self.random)

                                # Crossover between selected trees
                                crossover1 = Crossover.crossover(
                                    tree1=tree1, tree2=tree2, random=self.random)
                                crossover2 = Crossover.crossover(
                                    tree1=tree2, tree2=tree1, random=self.random)

                                # Mutation of new trees
                                mutation1 = crossover1
                                mutation2 = crossover2
                                if self.random.random() < mutation_probability:
                                    mutation1 = Mutation.mutation(root=crossover1, att_indexes=self.att_indexes,
                                                                  att_values=self.att_values, class_count=self.class_count,
                                                                  random=self.random)
                                if self.random.random() < mutation_probability:
                                    mutation2 = Mutation.mutation(root=crossover2, att_indexes=self.att_indexes,
                                                                  att_values=self.att_values, class_count=self.class_count,
                                                                  random=self.random)

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

            self._tree = population[0]
            return self
            
        except Exception as e:
            raise RuntimeError(f"Failed to fit GATreeRegressor: {str(e)}") from e