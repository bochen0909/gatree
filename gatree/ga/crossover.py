from gatree.tree.node import Node


class Crossover:
    """
    Class implementing the crossover operator for the genetic algorithm.
    """
    def crossover(tree1, tree2, random):
        """
        Crossover is a genetic operator used to combine the genetic information of two parent trees to generate new offspring. This enables exploration, which helps in creating diversity in the population and combining good traits from both parents.

        Pseudocode of the implementation:

        1. Copy the parent trees to avoid altering the originals.
        2. Randomly select a crossover point in each tree.
        3. Swap the subtrees at the selected points between the two trees.
        4. Return the new tree created from the crossover.

        Args:
            tree1 (Node): The first tree for crossover.
            tree2 (Node): The second tree for crossover.
            random (Random): Random number generator.

        Returns:
            Node: The new tree resulting from crossover.
        """
        if tree1 is None or tree2 is None:
            return tree1 if tree1 is not None else tree2
            
        n1 = Node.copy(tree1)
        n2 = Node.copy(tree2)
        
        if n1 is None or n2 is None:
            return n1 if n1 is not None else n2
            
        size1 = max(1, n1.max_depth())
        size2 = max(1, n2.max_depth())

        # Find crossover point in first tree
        attempts = 0
        max_attempts = 100
        while attempts < max_attempts:
            if n1 is None:
                break
            if (n1.left is None or random.randint(0, size1) == 0) and n1.parent is not None:
                break
            
            # Move to a child node
            if n1.left is not None and n1.right is not None:
                if random.choice([True, False]):
                    n1 = n1.left
                else:
                    n1 = n1.right
            elif n1.left is not None:
                n1 = n1.left
            elif n1.right is not None:
                n1 = n1.right
            else:
                break
            attempts += 1

        # Find crossover point in second tree
        attempts = 0
        while attempts < max_attempts:
            if n2 is None:
                break
            if (n2.left is None or random.randint(0, size2) == 0) and n2.parent is not None:
                break
                
            # Move to a child node
            if n2.left is not None and n2.right is not None:
                if random.choice([True, False]):
                    n2 = n2.left
                else:
                    n2 = n2.right
            elif n2.left is not None:
                n2 = n2.left
            elif n2.right is not None:
                n2 = n2.right
            else:
                break
            attempts += 1

        # Perform crossover if valid nodes found
        if n1 is not None and n2 is not None and n1.parent is not None:
            p = n1.parent
            if p.left == n1:
                p.set_left(n2)
            else:
                p.set_right(n2)
            return p.get_root()
        
        # Return original tree if crossover failed
        return Node.copy(tree1)
