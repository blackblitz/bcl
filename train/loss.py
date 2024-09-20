"""Loss functions."""

from jax import tree_util, vmap
import jax.numpy as jnp
import optax

from dataops import tree

from .probability import (
    gaussmix_output_kldiv_mc, gaussmix_output_kldiv_ub,
    gaussmix_params_kldiv_mc, gaussmix_params_kldiv_ub,
    gauss_output_kldiv, gauss_param, gauss_params_kldiv, gsgauss_param
)
from models import FinAct


def basic_loss(fin_act, precision, apply):
    """Return a basic loss function."""
    match fin_act:
        case FinAct.SIGMOID:

            def loss(params, xs, ys):
                return (
                    0.5 * precision * tree.dot(params, params)
                    + optax.sigmoid_binary_cross_entropy(
                        apply({'params': params}, xs)[:, 0], ys
                    ).sum()
                )

            return loss
        case FinAct.SOFTMAX:

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


def gvi_vfe(base, prior, beta):
    """Return the VFE function for Gaussian VI."""
    def loss(params, sample, xs, ys):
        param_sample = gauss_param(params, sample)
        return (
            vmap(base, in_axes=(0, None, None))(param_sample, xs, ys).mean()
            + beta * gauss_params_kldiv(params, prior)
        )

    return loss


def gmvi_vfe_mc(base, prior, beta):
    """Return the VFE function for Gaussian-mixture VI by MC integration."""
    def loss(params, sample, xs, ys):
        param_sample = gsgauss_param(params, sample)
        return (
            vmap(base, in_axes=(0, None, None))(param_sample, xs, ys).mean()
            + beta * gaussmix_params_kldiv_mc(param_sample, params, prior)
        )

    return loss


def gmvi_vfe_ub(base, prior, beta):
    """Return the VFE function for Gaussian-mixture VI by KL upper bound."""
    def loss(params, sample, xs, ys):
        param_sample = gsgauss_param(params, sample)
        return (
            vmap(base, in_axes=(0, None, None))(param_sample, xs, ys).mean()
            + beta * gaussmix_params_kldiv_ub(params, prior)
        )

    return loss


def gfsvi_vfe(base, prior, beta, apply):
    """Return the VFE function for Gaussian FSVI."""
    def loss(params, sample, xs1, ys1, xs2, ys2):
        param_sample = gauss_param(params, sample)
        return (
            vmap(base, in_axes=(0, None, None))(param_sample, xs1, ys1).mean()
            + beta * gauss_output_kldiv(params, prior, apply, xs2)
        )

    return loss


def gmfsvi_vfe_mc(key, base, prior, beta, apply):
    """Return the VFE function for Gaussian-mixture FSVI by MC integration."""

    def loss(params, sample, xs1, ys1, xs2, ys2):
        sample_size = len(tree_util.tree_leaves(sample)[0])
        param_sample = gsgauss_param(params, sample)
        return (
            vmap(base, in_axes=(0, None, None))(param_sample, xs1, ys1).mean()
            + beta * gaussmix_output_kldiv_mc(
                key, sample_size, params, prior, apply, xs2
            )
        )

    return loss


def gmfsvi_vfe_ub(base, prior, beta, apply):
    """Return the VFE function for Gaussian-mixture FSVI by KL upper bound."""
    def loss(params, sample, xs1, ys1, xs2, ys2):
        param_sample = gsgauss_param(params, sample)
        return (
            vmap(base, in_axes=(0, None, None))(param_sample, xs1, ys1).mean()
            + beta * gaussmix_output_kldiv_ub(params, prior, apply, xs2)
        )

    return loss
