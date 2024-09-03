"""Tree operations."""

from operator import add

from jax import random, tree_util


def normal(key, n, target):
    """Generate a Gaussian sample."""
    keys = random.split(key, len(tree_util.tree_leaves(target)))
    keys = tree_util.tree_unflatten(tree_util.tree_structure(target), keys)
    return tree_util.tree_map(
        lambda x, key: random.normal(key, (n, *x.shape)), target, keys
    )


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
