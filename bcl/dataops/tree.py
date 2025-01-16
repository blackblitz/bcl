"""Tree operations."""

from operator import add

from jax import random, tree_util
import jax.numpy as jnp


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


def full_like(target, value):
    """Return a tree with the same value for all elements."""
    return tree_util.tree_map(lambda x: jnp.full_like(x, value), target)


def gauss(key, target, loc=0.0, scale=1.0):
    """Generate a Gaussian tree."""
    keys = random.split(key, len(tree_util.tree_leaves(target)))
    keys = tree_util.tree_unflatten(tree_util.tree_structure(target), keys)
    return tree_util.tree_map(
        lambda x, key: loc + scale * random.normal(key, x.shape), target, keys
    )
