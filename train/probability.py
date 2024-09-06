"""Functions related to probability."""

import math

from jax import random, tree_util, vmap
from jax.nn import softmax, softplus
import jax.numpy as jnp


def get_mean_var(params, prior):
    """Compute the means and variances from the parameters."""
    return (
        params['mean'],
        tree_util.tree_map(lambda x: softplus(x) ** 2, params['msd']),
        prior['mean'],
        tree_util.tree_map(lambda x: softplus(x) ** 2, prior['msd'])
    )


def get_gauss_prior(precision, params):
    """Return the Gaussian prior parameters with fixed precision."""
    sd = math.sqrt(1 / precision)
    msd = sd + math.log(-math.expm1(-sd))
    return {
        'mean': tree_util.tree_map(jnp.zeros_like, params['mean']),
        'msd': tree_util.tree_map(
            lambda x: jnp.full_like(x, msd), params['msd']
        )
    }


def gauss_kldiv(mean, var, prior_mean, prior_var):
    """Compute the KL divergence of diagonal Gaussian PDFs."""
    return 0.5 * (
        (mean - prior_mean) ** 2 / prior_var - 1
        - jnp.log(var) + jnp.log(prior_var)
        + jnp.logaddexp(var, -prior_var)
    ).sum()


def gauss_sample(key, n, target):
    """Generate a Gaussian sample."""
    keys = random.split(key, len(tree_util.tree_leaves(target)))
    keys = tree_util.tree_unflatten(tree_util.tree_structure(target), keys)
    return {
        'gauss': tree_util.tree_map(
            lambda x, key: random.normal(key, (n, *x.shape)), target, keys
        )
    }


def gauss_gumbel_sample(key, n, m, target):
    """Generate Gaussian and Gumbel samples."""
    key1, key2 = random.split(key)
    return gauss_sample(
        key1, n, tree_util.tree_map(
            lambda x: jnp.repeat(jnp.expand_dims(x, 0), m, axis=0), target
        )
    ) | {'gumbel': random.gumbel(key2, (n, m))}


def gauss_param(params, sample):
    """Return a Gaussian parameterization function."""
    return tree_util.tree_map(
        lambda m, r, zs: m + softplus(r) * zs,
        params['mean'], params['msd'], sample['gauss']
    )


def gsgauss_param(params, sample):
    """Return a Gumbel-softmax-Gaussian-mixture parameterization function."""
    weight = softmax(1000 * (params['logit'] + sample['gumbel']))
    gauss = gauss_param(params, sample)
    return vmap(
        lambda w, g: tree_util.tree_map(
            lambda x: jnp.tensordot(x, w, axes=(0, 0)), g
        )
    )(weight, gauss)
