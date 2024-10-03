"""Stateless loss functions."""

from jax import flatten_util, tree_util, vmap
import jax.numpy as jnp
import optax

from dataops import tree
from models import NLL

from ..probability import (
    gaussmix_output_kldiv_mc, gaussmix_output_kldiv_ub,
    gaussmix_params_kldiv_mc, gaussmix_params_kldiv_ub,
    gauss_output_kldiv, gauss_param, gauss_params_kldiv, gsgauss_param
)


def get_nll(nll_enum):
    """Return the negative log-likelihood function."""
    match nll_enum:
        case NLL.SIGMOID_CROSS_ENTROPY:
            return sigmoid_cross_entropy
        case NLL.SOFTMAX_CROSS_ENTROPY:
            return softmax_cross_entropy


def sigmoid_cross_entropy(apply):
    """Return a sigmoid cross-entropy loss function."""
    def loss(params, xs, ys):
        return optax.sigmoid_binary_cross_entropy(
            apply({'params': params}, xs)[:, 0], ys
        ).sum()

    return loss


def softmax_cross_entropy(apply):
    """Return a softmax cross-entropy loss function."""
    def loss(params, xs, ys):
        return optax.softmax_cross_entropy_with_integer_labels(
            apply({'params': params}, xs), ys
        ).sum()

    return loss


def huber(apply):
    """Return a Huber loss function."""
    def loss(params, xs, ys):
        return optax.huber_loss(
            apply({'params': params}, xs)[:, 0], ys
        ).sum()

    return loss


def l2(apply):
    """Return an L2 loss function."""
    def loss(params, xs, ys):
        return optax.l2_loss(
            apply({'params': params}, xs)[:, 0], ys
        ).sum()

    return loss


def l2_reg(precision, nll):
    """Return an L2-regularized loss function."""
    def loss(params, xs, ys):
        return (
            0.5 * precision * tree.dot(params, params) + nll(params, xs, ys)
        )

    return loss


def concat(base):
    """Return a loss function by concatenating batches."""
    def loss(params, xs1, ys1, xs2, ys2):
        return base(
            params, jnp.concatenate([xs1, xs2]), jnp.concatenate([ys1, ys2])
        )

    return loss


def diag_quad_con(lambda_, minimum, hessian, nll):
    """Return a loss function for diagonal quadratic consolidation."""
    def loss(params, xs, ys):
        return (
            0.5 * lambda_ * tree.sum(
                tree_util.tree_map(
                    lambda h, p, m: h * (p - m) ** 2,
                    hessian, params, minimum
                )
            ) + nll(params, xs, ys)
        )

    return loss


def flat_quad_con(flat_minimum, flat_hessian, nll):
    """Return a loss function for flattened quadratic consolidation."""
    def loss(params, xs, ys):
        diff = flatten_util.ravel_pytree(params)[0] - flat_minimum
        return 0.5 * diff @ flat_hessian @ diff + nll(params, xs, ys)

    return loss


def neu_con(con_state, nll):
    """Return a loss function for neural consolidation."""
    def loss(params, xs, ys):
        return con_state.apply_fn(
            {'params': con_state.params},
            jnp.expand_dims(flatten_util.ravel_pytree(params)[0], 0)
        )[0, 0] + nll(params, xs, ys)

    return loss


def gvi_vfe(nll, prior, beta):
    """Return the VFE function for Gaussian VI."""
    def loss(params, sample, xs, ys):
        param_sample = gauss_param(params, sample)
        return (
            vmap(nll, in_axes=(0, None, None))(param_sample, xs, ys).mean()
            + beta * gauss_params_kldiv(params, prior)
        )

    return loss


def gmvi_vfe_mc(nll, prior, beta):
    """Return the VFE function for Gaussian-mixture VI by MC integration."""
    def loss(params, sample, xs, ys):
        param_sample = gsgauss_param(params, sample)
        return (
            vmap(nll, in_axes=(0, None, None))(param_sample, xs, ys).mean()
            + beta * gaussmix_params_kldiv_mc(param_sample, params, prior)
        )

    return loss


def gmvi_vfe_ub(nll, prior, beta):
    """Return the VFE function for Gaussian-mixture VI by KL upper bound."""
    def loss(params, sample, xs, ys):
        param_sample = gsgauss_param(params, sample)
        return (
            vmap(nll, in_axes=(0, None, None))(param_sample, xs, ys).mean()
            + beta * gaussmix_params_kldiv_ub(params, prior)
        )

    return loss


def gfsvi_vfe(nll, prior, beta, apply):
    """Return the VFE function for Gaussian FSVI."""
    def loss(params, sample, xs1, ys1, xs2, ys2):
        param_sample = gauss_param(params, sample)
        return (
            vmap(nll, in_axes=(0, None, None))(param_sample, xs1, ys1).mean()
            + beta * gauss_output_kldiv(params, prior, apply, xs2)
        )

    return loss


def gmfsvi_vfe_mc(key, nll, prior, beta, apply):
    """Return the VFE function for Gaussian-mixture FSVI by MC integration."""

    def loss(params, sample, xs1, ys1, xs2, ys2):
        sample_size = len(tree_util.tree_leaves(sample)[0])
        param_sample = gsgauss_param(params, sample)
        return (
            vmap(nll, in_axes=(0, None, None))(param_sample, xs1, ys1).mean()
            + beta * gaussmix_output_kldiv_mc(
                key, sample_size, params, prior, apply, xs2
            )
        )

    return loss


def gmfsvi_vfe_ub(nll, prior, beta, apply):
    """Return the VFE function for Gaussian-mixture FSVI by KL upper bound."""
    def loss(params, sample, xs1, ys1, xs2, ys2):
        param_sample = gsgauss_param(params, sample)
        return (
            vmap(nll, in_axes=(0, None, None))(param_sample, xs1, ys1).mean()
            + beta * gaussmix_output_kldiv_ub(params, prior, apply, xs2)
        )

    return loss
