"""Loss functions."""

from jax import flatten_util, tree_util, vmap
import jax.numpy as jnp
import optax

from dataops import tree
from models import NLL

from .vi import gauss, gaussmix, t


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


def flat_quad_con(lambda_, flat_minimum, flat_hessian, nll):
    """Return a loss function for flattened quadratic consolidation."""
    def loss(params, xs, ys):
        diff = flatten_util.ravel_pytree(params)[0] - flat_minimum
        return (
            0.5 * lambda_ * diff @ flat_hessian @ diff + nll(params, xs, ys)
        )

    return loss


def neu_con(con_state, nll):
    """Return a loss function for neural consolidation."""
    def loss(params, xs, ys):
        return con_state.apply_fn(
            {'params': con_state.params},
            jnp.expand_dims(flatten_util.ravel_pytree(params)[0], 0)
        )[0, 0] + nll(params, xs, ys)

    return loss


def gpvfe_cf(nll, beta, prior):
    """Return the closed-form Gaussian parameter-space VFE."""
    def loss(params, sample, xs, ys):
        q = gauss.get_param(params)
        p = gauss.get_param(prior)
        param_sample = gauss.transform(q, sample)
        return (
            vmap(nll, in_axes=(0, None, None))(param_sample, xs, ys).mean()
            + beta * gauss.kldiv_cf(q, p)
        )

    return loss


def gpvfe_mc(nll, beta, prior):
    """Return the Monte Carlo Gaussian parameter-space VFE."""
    def loss(params, sample, xs, ys):
        q = gauss.get_param(params)
        p = gauss.get_param(prior)
        param_sample = gauss.transform(q, sample)
        return (
            vmap(nll, in_axes=(0, None, None))(param_sample, xs, ys).mean()
            + beta * gauss.kldiv_mc(param_sample, q, p)
        )

    return loss


def gfvfe_cf(nll, beta, prior, apply):
    """Return the closed-form Gaussian function-space VFE."""
    def loss(params, param_sample, output_sample, xs, ys, ind_xs):
        param_sample = gauss.transform(gauss.get_param(params), param_sample)
        q = gauss.get_output(params, apply, ind_xs)
        p = gauss.get_output(prior, apply, ind_xs)
        output_sample = gauss.transform(q, output_sample)
        return (
            vmap(nll, in_axes=(0, None, None))(param_sample, xs, ys).mean()
            + beta * gauss.kldiv_cf(q, p)
        )

    return loss


def gfvfe_mc(nll, beta, prior, apply):
    """Return the Monte Carlo Gaussian function-space VFE."""
    def loss(params, param_sample, output_sample, xs, ys, ind_xs):
        param_sample = gauss.transform(gauss.get_param(params), param_sample)
        q = gauss.get_output(params, apply, ind_xs)
        p = gauss.get_output(prior, apply, ind_xs)
        output_sample = gauss.transform(q, output_sample)
        return (
            vmap(nll, in_axes=(0, None, None))(param_sample, xs, ys).mean()
            + beta * gauss.kldiv_mc(output_sample, q, p)
        )

    return loss


def tpvfe_mc(nll, beta, prior, df):
    """Return the Monte Carlo t parameter-space VFE."""
    def loss(params, sample, xs, ys):
        q = t.get_param(params, df)
        p = t.get_param(prior, df)
        param_sample = t.transform(q, sample)
        return (
            vmap(nll, in_axes=(0, None, None))(param_sample, xs, ys).mean()
            + beta * t.kldiv_mc(param_sample, q, p)
        )

    return loss


def tfvfe_mc(nll, beta, prior, apply, df):
    """Return the Monte Carlo t function-space VFE."""
    def loss(params, param_sample, output_sample, xs, ys, ind_xs):
        param_sample = t.transform(t.get_param(params, df), param_sample)
        q = t.get_output(params, df, apply, ind_xs)
        p = t.get_output(prior, df, apply, ind_xs)
        output_sample = t.transform(q, output_sample)
        return (
            vmap(nll, in_axes=(0, None, None))(param_sample, xs, ys).mean()
            + beta * t.kldiv_mc(output_sample, q, p)
        )

    return loss


def gmpvfe_ub(nll, beta, prior):
    """Return the upper-bound Gaussian mixture parameter-space VFE."""
    def loss(params, sample, xs, ys):
        q = gaussmix.get_param(params)
        p = gaussmix.get_param(prior)
        param_sample = gaussmix.transform(q, sample)
        return (
            vmap(nll, in_axes=(0, None, None))(param_sample, xs, ys).mean()
            + beta * gaussmix.kldiv_ub(q, p)
        )

    return loss


def gmpvfe_mc(nll, beta, prior):
    """Return the Monte Carlo Gaussian mixture parameter-space VFE."""
    def loss(params, sample, xs, ys):
        q = gaussmix.get_param(params)
        p = gaussmix.get_param(prior)
        param_sample = gaussmix.transform(q, sample)
        return (
            vmap(nll, in_axes=(0, None, None))(param_sample, xs, ys).mean()
            + beta * gaussmix.kldiv_mc(param_sample, q, p)
        )

    return loss


def gmfvfe_ub(nll, beta, prior, apply):
    """Return the upper-bound Gaussian mixture function-space VFE."""
    def loss(params, param_sample, output_sample, xs, ys, ind_xs):
        param_sample = gaussmix.transform(
            gaussmix.get_param(params), param_sample
        )
        q = gaussmix.get_output(params, apply, ind_xs)
        p = gaussmix.get_output(prior, apply, ind_xs)
        output_sample = gaussmix.transform(q, output_sample)
        return (
            vmap(nll, in_axes=(0, None, None))(param_sample, xs, ys).mean()
            + beta * gaussmix.kldiv_cf(q, p)
        )

    return loss


def gmfvfe_mc(nll, beta, prior, apply):
    """Return the Monte Carlo Gaussian mixture function-space VFE."""
    def loss(params, param_sample, output_sample, xs, ys, ind_xs):
        param_sample = gaussmix.transform(
            gaussmix.get_param(params), param_sample
        )
        q = gaussmix.get_output(params, apply, ind_xs)
        p = gaussmix.get_output(prior, apply, ind_xs)
        output_sample = gaussmix.transform(q, output_sample)
        return (
            vmap(nll, in_axes=(0, None, None))(param_sample, xs, ys).mean()
            + beta * gaussmix.kldiv_mc(output_sample, q, p)
        )

    return loss
