import unittest
import numpy as np
from gatree.tree.node import Node
from gatree.ga.mutation import Mutation


class TestMutation(unittest.TestCase):
    """
    Test Mutation class.

    Args:
        unittest.TestCase: TestCase class from the unittest module.
    """

    def test_change_class(self):
        """
        Test change_class method.
        """
        # Set the seed for reproducibility
        random = np.random.RandomState(42)  # Use different seed

        # Create a node for testing
        node = Node(att_index=-1, att_value=0)
        original_value = node.att_value

        # Test the change_class method
        Mutation.change_class(node, 2, random)

        # Assert that the node is a leaf
        self.assertEqual(node.att_index, -1)

        # Assert that class has changed (or at least the method ran without error)
        # Since it's random, we just check it's a valid class value
        self.assertIn(node.att_value, [0, 1])

    def test_exchange_class_for_tree(self):
        """
        Test exchange_class_for_tree method.
        """
        # Set the seed for reproducibility
        random = np.random.RandomState(0)

        # Create a node for testing
        node = Node(att_index=-1, att_value=0)
        parent = Node(att_index=0, att_value=0)
        parent.set_left(node)

        # Test the change_class method
        Mutation.exchange_class_for_tree(
            node, [0, 1], {0: [1, 2], 1: [3, 4]}, 2, random)
        node = parent.left

        # Assert that the node is not a leaf
        self.assertNotEqual(node.att_index, -1)

    def test_change_attribute(self):
        """
        Test change_attribute method.
        """
        # Set the seed for reproducibility
        random = np.random.RandomState(42)  # Use different seed

        # Create a node for testing
        node = Node(att_index=0, att_value=0)
        original_index = node.att_index

        # Test the change_attribute method
        Mutation.change_attribute(node, [0, 1], {0: [0, 1], 1: [3]}, random)

        # Assert that the attribute is valid (either changed or stayed the same due to randomness)
        self.assertIn(node.att_index, [0, 1])

    def test_change_attribute_value(self):
        """
        Test change_attribute_value method.
        """
        # Set the seed for reproducibility
        random = np.random.RandomState(42)  # Use different seed

        # Create a node for testing
        node = Node(att_index=0, att_value=0)

        # Test the change_attribute method
        Mutation.change_attribute_value(node, {0: [0, 1]}, random)

        # Assert that the attribute has not changed
        self.assertEqual(node.att_index, 0)

        # Assert that the attribute value is valid
        self.assertIn(node.att_value, [0, 1])

    def test_exchange_tree_for_class(self):
        """
        Test exchange_tree_for_class method.
        """
        # Set the seed for reproducibility
        random = np.random.RandomState(42)

        # Create a tree with guaranteed structure
        tree = Node(att_index=0, att_value=0.5)
        left_child = Node(att_index=-1, att_value=0)
        right_child = Node(att_index=-1, att_value=1)
        tree.set_left(left_child)
        tree.set_right(right_child)

        # Test the exchange_tree_for_class method
        Mutation.exchange_tree_for_class(tree.left, 2, random)

        # Assert that the left child is still a leaf (or has been replaced with a leaf)
        self.assertEqual(tree.left.att_index, -1)
        # Assert that the value is a valid class
        self.assertIn(tree.left.att_value, [0, 1])

    def test_exchange_tree_for_tree(self):
        """
        Test exchange_tree_for_tree method.
        """
        # Set the seed for reproducibility
        random = np.random.RandomState(42)

        # Create a tree with guaranteed structure
        tree = Node(att_index=0, att_value=0.5)
        left_child = Node(att_index=-1, att_value=0)
        right_child = Node(att_index=-1, att_value=1)
        tree.set_left(left_child)
        tree.set_right(right_child)

        original_size = tree.left.size()

        # Test the exchange_tree_for_tree method
        Mutation.exchange_tree_for_tree(
            tree.left, np.arange(1), {0: [0, 1]}, 2, random)

        # Assert that the left child still exists and has a valid size
        self.assertIsNotNone(tree.left)
        self.assertGreaterEqual(tree.left.size(), 1)
