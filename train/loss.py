"""Loss functions."""

from operator import add, itemgetter

import numpy as np
from jax import grad, jit, tree_util


def make_loss(state, loss_point, multi=True):
    """Make a loss function."""
    get = itemgetter(np.s_[:] if multi else np.s_[:, 0])
    return jit(
        lambda params, x, y: loss_point(
            get(state.apply_fn({'params': params}, x)), y
        ).sum()
    )


def make_loss_reg(precision, loss_basic):
    """Make a loss function with a regularization term."""
    return jit(
        lambda params, x, y:
        tree_util.tree_reduce(
            add,
            tree_util.tree_map(
                lambda x: 0.5 * (precision * x ** 2).sum(), params
            )
        ) + loss_basic(params, x, y)
    )


def make_step(loss):
    """Make a gradient-descent step function for a loss function."""
    return jit(
        lambda state, x, y: state.apply_gradients(
            grads=grad(loss)(state.params, x, y)
        )
    )
