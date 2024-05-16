"""Random module."""

from jax import random, tree_util


def gausstree(key, n, tree):
    """Generate a Gaussian sample of a pytree."""
    keys = random.split(key, len(tree_util.tree_leaves(tree)))
    keys = tree_util.tree_unflatten(tree_util.tree_structure(tree), keys)
    return tree_util.tree_map(
        lambda x, key: random.normal(key, (n, *x.shape)), tree, keys
    )
