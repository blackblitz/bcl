"""Student's t KL divergence."""

import math

from jax import nn, random, tree_util, vmap
import jax.numpy as jnp
from jax.scipy.special import gammaln

from ..neural_tangents.extended import empirical_ntk

from ....dataops import tree


def get_prior(invscale, params):
    """Return the t prior parameters with fixed inverse scale."""
    rs = math.sqrt(1 / invscale)
    mrs = rs + math.log(-math.expm1(-rs))
    return {
        'loc': tree_util.tree_map(jnp.zeros_like, params['loc']),
        'ms': tree_util.tree_map(
            lambda x: jnp.full_like(x, mrs), params['ms']
        )
    }


def logpdf(value, loc, scale, df):
    """Compute the Student's t log density."""
    size = tree.size(value)
    return gammaln(0.5 * (df + size)) - gammaln(0.5 * df) - 0.5 * (
        size * jnp.log(df * jnp.pi)
        + tree.sum(tree_util.tree_map(jnp.log, scale))
        + (df + size) * jnp.log(1 + 1 / df * tree.sum(tree_util.tree_map(
            lambda x, m, s: ((x - m) ** 2 / s), value, loc, scale
        )))
    )


def kldiv_mc(sample, q, p):
    """Compute the Monte Carlo Gaussian KL divergence."""
    return vmap(
        lambda x:
        logpdf(x, q['loc'], q['scale'], q['df'])
        - logpdf(x, p['loc'], p['scale'], p['df'])
    )(sample).mean()


def get_param(params, df):
    """Compute the parameters of the parameters."""
    return {
        'loc': params['loc'],
        'scale': tree_util.tree_map(
            lambda x: nn.softplus(x) ** 2, params['ms']
        ),
        'df': df
    }


def get_output(params, df, apply, xs):
    """Compute the parameters of the outputs."""
    param = get_param(params, df)
    loc = apply({'params': param['loc']}, xs)
    scale = empirical_ntk(apply, (), (0, 1), 0)(
        xs, None, {'params': param['loc']}, {'params': param['scale']}
    )
    return {'loc': loc, 'scale': scale, 'df': df}


def sample(key, n, target, df):
    """Generate Gaussian and gamma samples."""
    key1, key2 = random.split(key)
    return {
        'gauss': vmap(
            lambda x: tree.gauss(x, target)
        )(random.split(key1, num=n)),
        'gamma': 2 / df * random.gamma(key2, df / 2, shape=(n,))
    }


def transform(q, sample):
    """Transform a Student's t sample."""
    return vmap(
        lambda z, u: tree_util.tree_map(
            lambda ml, sl, zl: ml + jnp.sqrt(sl / u) * zl,
            q['loc'], q['scale'], z
        )
    )(sample['gauss'], sample['gamma'])
