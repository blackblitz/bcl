"""Tree operations."""

from operator import add

from jax import random, tree_util


def size(tree):
    """Compute the size of a tree."""
    return tree_util.tree_reduce(
         add, tree_util.tree_map(lambda x: x.size, tree)
    )


def sum(tree):
    """Compute the sum of a tree."""
    return tree_util.tree_reduce(
        add, tree_util.tree_map(lambda x: x.sum(), tree)
    )


def dot(tree1, tree2):
    """Compute the dot product of two trees."""
    return tree_util.tree_reduce(
        add, tree_util.tree_map(lambda x, y: (x * y).sum(), tree1, tree2)
    )
