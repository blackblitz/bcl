"""Loss functions."""

import optax

from dataops import tree
from models import NLL


def get_nll(nll_enum):
    """Return the negative log-likelihood function."""
    match nll_enum:
        case NLL.SIGMOID_CROSS_ENTROPY:
            return sigmoid_cross_entropy
        case NLL.SOFTMAX_CROSS_ENTROPY:
            return softmax_cross_entropy


def sigmoid_cross_entropy(apply, **kwargs):
    """Return a sigmoid cross-entropy loss function."""
    def loss(params, var, rngs, xs, ys):
        out, var = apply({'params': params} | var, xs, rngs=rngs, **kwargs)
        return optax.sigmoid_binary_cross_entropy(out[:, 0], ys).sum(), var

    return loss


def softmax_cross_entropy(apply, **kwargs):
    """Return a softmax cross-entropy loss function."""
    def loss(params, var, rngs, xs, ys):
        out, var = apply({'params': params} | var, xs, rngs=rngs, **kwargs)
        return (
            optax.softmax_cross_entropy_with_integer_labels(out, ys).sum(),
            var
        )

    return loss


def l2_reg(precision, nll):
    """Return an L2-regularized loss function."""
    def loss(params, var, rngs, xs, ys):
        loss_val, var = nll(params, var, rngs, xs, ys)
        return (
            0.5 * precision * tree.dot(params, params) + loss_val,
            var
        )

    return loss
