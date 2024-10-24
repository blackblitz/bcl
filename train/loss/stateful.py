"""Stateful loss functions."""

from jax import flatten_util, tree_util, vmap
import jax.numpy as jnp
import optax

from dataops import tree
from models import NLL

from ..probability import (
    gaussmix_output_kldiv_mc, gaussmix_output_kldiv_ub,
    gaussmix_params_kldiv_mc, gaussmix_params_kldiv_ub,
    gauss_output_kldiv, gauss_param, gauss_params_kldiv, gsgauss_param,
    t_output_kldiv_mc, t_param, t_params_kldiv_mc
)


def out_nll(nll):
    """Return the negative log-likelihood as a function of outputs."""
    match nll:
        case NLL.HUBER:
            return lambda out, ys: optax.huber_loss(out[:, 0], ys).sum()
        case NLL.L2:
            return lambda out, ys: optax.l2_loss(out[:, 0], ys).sum()
        case NLL.SIGMOID_CROSS_ENTROPY:
            return lambda out, ys: optax.sigmoid_binary_cross_entropy(
                out[:, 0], ys
            ).sum()
        case NLL.SOFTMAX_CROSS_ENTROPY:
            return (
                lambda out, ys:
                optax.softmax_cross_entropy_with_integer_labels(
                    out, ys
                ).sum()
            )


def param_nll(out_nll, apply, train: bool):
    """Return the negative log-likelihood as a function of parameters."""
    def loss(params, mut, rngs, xs, ys):
        out, mut = apply(
            {'params': params} | mut,
            xs, train,
            rngs=rngs, mutable=list(mut.keys())
        )
        return out_nll(out, ys), mut

    return loss


def l2_reg(precision, param_nll):
    """Return an L2-regularized loss function with mutable variables."""
    def loss(params, mut, rngs, xs, ys):
        nll, mut = param_nll(params, mut, rngs, xs, ys)
        return (
            0.5 * precision * tree.dot(params, params) + nll, mut
        )

    return loss


def concat(base):
    """Return a loss function by concatenating batches."""
    def loss(params, mut, rngs, xs1, ys1, xs2, ys2):
        return base(
            params, mut, rngs,
            jnp.concatenate([xs1, xs2]), jnp.concatenate([ys1, ys2])
        )

    return loss


def diag_quad_con(lambda_, minimum, hessian, param_nll):
    """Return a loss function for diagonal quadratic consolidation."""
    def loss(params, mut, rngs, xs, ys):
        nll, mut = param_nll(params, mut, rngs, xs, ys)
        return (
            0.5 * lambda_ * tree.sum(
                tree_util.tree_map(
                    lambda h, p, m: h * (p - m) ** 2,
                    hessian, params, minimum
                )
            ) + nll, mut
        )

    return loss


def flat_quad_con(lambda_, flat_minimum, flat_hessian, param_nll):
    """Return a loss function for flattened quadratic consolidation."""
    def loss(params, mut, rngs, xs, ys):
        nll, mut = param_nll(params, mut, rngs, xs, ys)
        diff = flatten_util.ravel_pytree(params)[0] - flat_minimum
        return (
            0.5 * lambda_ * diff @ flat_hessian @ diff + nll, mut
        )

    return loss


def neu_con(con_state, param_nll):
    """Return a loss function for neural consolidation."""
    def loss(params, mut, rngs, xs, ys):
        nll, mut = param_nll(params, mut, rngs, xs, ys)
        return con_state.apply_fn(
            {'params': con_state.params},
            jnp.expand_dims(flatten_util.ravel_pytree(params)[0], 0)
        )[0, 0] + nll, mut

    return loss


def psvfe(param_nll, beta, kldiv):
    """Return the parameter space variational free energy loss function."""
    def loss(params, sample, mut, rngs, xs, ys):
        param_sample = gauss_param(params, sample)
        nll, mut = vmap(
            param_nll, in_axes=(0, None, None, None, None)
        )(param_sample, mut, rngs, xs, ys)
        return (
            nll.mean() + beta * kldiv(params),
            tree_util.tree_map(lambda x: x.mean(axis=0), mut)
        )

    return loss


def fsvfe(param_nll, beta, kldiv):
    """Return the VFE function for Gaussian FSVI."""
    def loss(params, sample, mut, rngs, xs, ys, ind_xs):
        param_sample = gauss_param(params, sample)
        nll, mut = vmap(
            param_nll, in_axes=(0, None, None, None, None)
        )(param_sample, mut, rngs, xs, ys)
        return (
            nll.mean() + beta * kldiv(params, ind_xs),
            tree_util.tree_map(lambda x: x.mean(axis=0), mut)
        )

    return loss
