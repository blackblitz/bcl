"""Loss functions."""

from operator import add

from jax import jit, tree_util, vmap
from jax.nn import softmax
import jax.numpy as jnp
from jax.scipy.special import rel_entr
import optax

from . import tree
from .neural_tangents.extended import empirical_ntk
from .probability import gauss_kldiv, gauss_param, get_mean_var, gsgauss_param


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


def concat_loss(base):
    """Return a loss function by concatenating batches."""
    def loss(params, xs1, ys1, xs2, ys2):
        return base(
            params, jnp.concatenate([xs1, xs2]), jnp.concatenate([ys1, ys2])
        )

    return loss


def expected_loss(base, parameterize, sample):
    """Return an expected loss function."""
    def loss(params, xs, ys):
        return vmap(
            base, in_axes=(0, None, None)
        )(parameterize(params, sample), xs, ys).mean()

    return loss


def gvi_kldiv(params, prior):
    """Compute the KL divergence for Gaussian VI."""
    return tree_util.tree_reduce(
        add, tree_util.tree_map(
            gauss_kldiv, *get_mean_var(params, prior)
        )
    )


def gvi_vfe(base, sample, prior, beta):
    """Return the variational free energy function for Gaussian VI."""
    def loss(params, xs, ys):
        return (
            expected_loss(base, gauss_param, sample)(params, xs, ys)
            + beta * gvi_kldiv(params, prior)
        )

    return loss


def gmvi_kldiv(params, prior):
    """Compute a KL upper bound for Gaussian-mixture VI."""
    weight = softmax(params['logit'])
    prior_weight = softmax(prior['logit'])
    cat_kldiv = rel_entr(weight, prior_weight).sum()
    gauss_kldiv = vmap(gvi_kldiv)(params, prior)
    return cat_kldiv + weight @ gauss_kldiv


def gmvi_vfe(base, sample, prior, beta):
    """Return the variational free energy function for Gaussian-mixture VI."""
    def loss(params, xs, ys):
        return (
            expected_loss(base, gsgauss_param, sample)(params, xs, ys)
            + beta * gmvi_kldiv(params, prior)
        )

    return loss


def gfsvi_kldiv(params, prior, apply, xs):
    """Compute the KL divergence for Gaussian FSVI."""
    mean, var, prior_mean, prior_var = get_mean_var(params, prior)
    func_mean = apply({'params': mean}, xs)
    prior_func_mean = apply({'params': prior_mean}, xs)
    get_func_var = empirical_ntk(apply, (), (0, 1), 0)
    func_var = get_func_var(xs, None, {'params': mean}, var)
    prior_func_var = get_func_var(xs, None, {'params': prior_mean}, prior_var)
    return gauss_kldiv(func_mean, func_var, prior_func_mean, prior_func_var)


def gfsvi_vfe(base, sample, prior, beta, apply):
    """Return the variational free energy function for Gaussian FSVI."""
    def loss(params, xs1, ys1, xs2, ys2):
        return (
            expected_loss(base, gauss_param(sample))(params, xs1, ys1)
            + beta * gfsvi_kldiv(params, prior, apply, xs2)
        )
    
    return loss
