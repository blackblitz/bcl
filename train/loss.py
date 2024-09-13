"""Loss functions."""

from operator import add, sub

from jax import jvp, tree_util, vmap
from jax.nn import softmax
import jax.numpy as jnp
from jax.scipy.special import rel_entr
import optax

from dataops import tree

from .neural_tangents.extended import empirical_ntk
from .probability import gauss_kldiv, gauss_param, get_mean_var, gsgauss_param
from .trainer import NNType


def basic_loss(nntype, precision, apply):
    """Return a basic loss function."""
    nntype = NNType[nntype]
    match nntype:
        case NNType.SIGMOID:

            def loss(params, xs, ys):
                return (
                    0.5 * precision * tree.dot(params, params)
                    + optax.sigmoid_binary_cross_entropy(
                        apply({'params': params}, xs)[:, 0], ys
                    ).sum()
                )

            return loss
        case NNType.SOFTMAX:

            def loss(params, xs, ys):
                return (
                    0.5 * precision * tree.dot(params, params)
                    + optax.softmax_cross_entropy_with_integer_labels(
                        apply({'params': params}, xs), ys
                    ).sum()
                )

            return loss


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
    """Return the VFE function for Gaussian VI."""
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
    cat_kldiv_val = rel_entr(weight, prior_weight).sum()
    gauss_kldiv_val = vmap(gvi_kldiv)(params, prior)
    return cat_kldiv_val + weight @ gauss_kldiv_val


def gmvi_vfe(base, sample, prior, beta):
    """Return the VFE function for Gaussian-mixture VI."""
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
    func_var = get_func_var(
        xs, None, {'params': mean}, {'params': var}
    )
    prior_func_var = get_func_var(
        xs, None, {'params': prior_mean}, {'params': prior_var}
    )
    return gauss_kldiv(func_mean, func_var, prior_func_mean, prior_func_var)


def gfsvi_vfe(base, sample, prior, beta, apply):
    """Return the VFE function for Gaussian FSVI."""
    def loss(params, xs1, ys1, xs2, ys2):
        return (
            expected_loss(base, gauss_param, sample)(params, xs1, ys1)
            + beta * gfsvi_kldiv(params, prior, apply, xs2)
        )

    return loss


def gmfsvi_kldiv(params, prior, apply, xs):
    """Compute the KL divergence for Gaussian-mixture FSVI."""
    mean, var, prior_mean, prior_var = get_mean_var(params, prior)
    center = tree_util.tree_map(lambda x: x.mean(axis=0), mean)
    prior_center = tree_util.tree_map(lambda x: x.mean(axis=0), prior_mean)

    def get_func_mean(center, mean):
        """Compute the function-space mean."""
        return vmap(
            lambda m:
            apply({'params': center}, xs)
            + jvp(
                lambda params: apply(params, xs),
                ({'params': center},),
                ({'params': tree_util.tree_map(sub, m, center)},)
            )[1]
        )(mean)

    func_mean = get_func_mean(center, mean)
    prior_func_mean = get_func_mean(prior_center, prior_mean)
    get_func_var = empirical_ntk(apply, (), (0, 1), 0)
    func_var = vmap(get_func_var, in_axes=(None, None, None, 0))(
        xs, None, {'params': center}, {'params': var}
    )
    prior_func_var = vmap(get_func_var, in_axes=(None, None, None, 0))(
        xs, None, {'params': prior_center}, {'params': prior_var}
    )
    weight = softmax(params['logit'])
    prior_weight = softmax(prior['logit'])
    cat_kldiv_val = rel_entr(weight, prior_weight).sum()
    gauss_kldiv_val = vmap(gauss_kldiv)(
        func_mean, func_var, prior_func_mean, prior_func_var
    )
    return cat_kldiv_val + weight @ gauss_kldiv_val


def gmfsvi_vfe(base, sample, prior, beta, apply):
    """Return the VFE function for Gaussian-mixture FSVI."""
    def loss(params, xs1, ys1, xs2, ys2):
        return (
            expected_loss(base, gsgauss_param, sample)(params, xs1, ys1)
            + beta * gmfsvi_kldiv(params, prior, apply, xs2)
        )

    return loss
