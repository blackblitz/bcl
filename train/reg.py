"""Regular training."""

from operator import add

from jax import jit, tree_util


def make_loss(state, hyperparams, loss_basic):
    return jit(
        lambda params, x, y: 
        tree_util.tree_reduce(
            add,
            tree_util.tree_map(
                lambda x: 0.5 * (hyperparams['precision'] * x ** 2).sum(),
                params
            )
        ) + loss_basic(params, x, y)
    )


def update_hyperparams(state, hyperparams, loss_basic, batches):
    pass
