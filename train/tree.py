"""Random module."""

from operator import add

import jax.numpy as jnp
from jax import random, tree_util


def tree_gauss(key, n, tree):
    """Generate a Gaussian sample of a pytree."""
    keys = random.split(key, len(tree_util.tree_leaves(tree)))
    keys = tree_util.tree_unflatten(tree_util.tree_structure(tree), keys)
    return tree_util.tree_map(
        lambda x, key: random.normal(key, (n, *x.shape)), tree, keys
    )


def tree_size(tree):
    """Compute the size of a pytree."""
    return tree_util.tree_reduce(
         add, tree_util.tree_map(lambda x: x.size, tree)
    )


def tree_sum(tree):
    """Compute the sum of a pytree."""
    return tree_util.tree_reduce(
        add, tree_util.tree_map(jnp.sum, tree)
    )


def tree_dot(tree1, tree2):
    """Compute the dot product of two pytrees."""
    return tree_util.tree_reduce(
        add,
        tree_util.tree_map(lambda a, b: (a * b).sum(), tree1, tree2)
    )
