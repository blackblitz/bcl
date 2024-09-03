"""Loss functions."""

from operator import add

import jax.numpy as jnp
import optax

from . import tree


def sigmoid_ce(precision, apply):
    """Return a loss function for sigmoid cross-entropy."""
    def loss(params, xs, ys):
        return (
            0.5 * precision * tree.dot(params, params)
            + optax.sigmoid_binary_cross_entropy(
                apply({'params': params}, xs)[:, 0], ys
            ).sum()
        )

    return loss


def softmax_ce(precision, apply):
    """Return a loss function for softmax cross-entropy."""
    def loss(params, xs, ys):
        return (
            0.5 * precision * tree.dot(params, params)
            + optax.softmax_cross_entropy_with_integer_labels(
                apply({'params': params}, xs), ys
            ).sum()
        )

    return loss


def huber(precision, apply):
    """Return a loss function for Huber."""
    def loss(params, xs, ys):
        return (
            0.5 * precision * tree.dot(params, params)
            + optax.huber_loss(
                apply({'params': params}, xs)[:, 0], ys
            ).sum()
        )

    return loss


def concat_loss(primal):
    """Return a loss function by concatenating batches."""
    def loss(params, xs1, ys1, xs2, ys2):
        return primal(
            params, jnp.concatenate([xs1, xs2]), jnp.concatenate([ys1, ys2])
        )

    return loss
