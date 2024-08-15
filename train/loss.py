"""Loss functions."""

from operator import add

from jax import jit, tree_util
import optax


def sigmoid(apply):
    """Return a loss function for sigmoid cross-entropy."""
    return jit(
        lambda params, x, y: optax.sigmoid_binary_cross_entropy(
            apply({'params': params}, x)[:, 0], y
        ).sum()
    )


def softmax(apply):
    """Return a loss function for softmax cross-entropy."""
    return jit(
        lambda params, x, y: optax.softmax_cross_entropy_with_integer_labels(
            apply({'params': params}, x), y
        ).sum()
    )


def huber(apply):
    """Return a loss function for Huber."""
    return jit(
        lambda params, x, y: optax.huber_loss(
            apply({'params': params}, x)[:, 0], y
        ).sum()
    )


def reg(precision, basic_loss_fn):
    """Make a loss function with a regularization term."""
    return jit(
        lambda params, x, y:
        tree_util.tree_reduce(
            add,
            tree_util.tree_map(
                lambda x: 0.5 * (precision * x ** 2).sum(), params
            )
        ) + basic_loss_fn(params, x, y)
    )
