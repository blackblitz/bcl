"""Gaussian KL divergence."""

import math

from jax import nn, random, tree_util, vmap
import jax.numpy as jnp

from ..neural_tangents.extended import empirical_ntk

from ....dataops import tree


def get_prior(precision, params):
    """Return the Gaussian prior parameters with fixed precision."""
    sd = math.sqrt(1 / precision)
    msd = sd + math.log(-math.expm1(-sd))
    return {
        'mean': tree_util.tree_map(jnp.zeros_like, params['mean']),
        'msd': tree_util.tree_map(
            lambda x: jnp.full_like(x, msd), params['msd']
        )
    }


def kldiv_cf(q, p):
    """Compute the closed-form Gaussian KL divergence."""
    return tree.sum(tree_util.tree_map(
        lambda m, v, pm, pv: 0.5 * (
            -jnp.log(v / pv) + (v + (m - pm) ** 2) / pv - 1
        ),
        q['mean'], q['var'], p['mean'], p['var']
    ))


def logpdf(value, mean, var):
    """Compute the Gaussian log density."""
    size = tree.size(value)
    return -0.5 * (
        size * jnp.log(2 * jnp.pi)
        + tree.sum(tree_util.tree_map(jnp.log, var))
        + tree.sum(tree_util.tree_map(
            lambda x, m, v: (x - m) ** 2 / v, value, mean, var
        ))
    )


def kldiv_mc(sample, q, p):
    """Compute the Monte Carlo Gaussian KL divergence."""
    return vmap(
        lambda x:
        logpdf(x, q['mean'], q['var'])
        - logpdf(x, p['mean'], p['var'])
    )(sample).mean()


def get_param(params):
    """Compute the parameters of the parameters."""
    return {
        'mean': params['mean'],
        'var': tree_util.tree_map(lambda x: nn.softplus(x) ** 2, params['msd'])
    }


def get_output(params, apply, xs):
    """Compute the parameters of the outputs."""
    param = get_param(params)
    mean = apply({'params': param['mean']}, xs)
    var = empirical_ntk(apply, (), (0, 1), 0)(
        xs, None, {'params': param['mean']}, {'params': param['var']}
    )
    return {'mean': mean, 'var': var}


def sample(key, n, target):
    """Generate a Gaussian sample."""
    return {
        'gauss': vmap(
            lambda x: tree.gauss(x, target)
        )(random.split(key, num=n))
    }


def transform(q, sample):
    """Transform a Gaussian sample."""
    return vmap(
        lambda z: tree_util.tree_map(
            lambda ml, vl, zl: ml + jnp.sqrt(vl) * zl,
            q['mean'], q['var'], z
        )
    )(sample['gauss'])
