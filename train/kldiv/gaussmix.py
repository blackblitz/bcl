"""Gaussian mixture KL divergence."""

from operator import sub

from jax import jvp, nn, random, tree_util, vmap
import jax.numpy as jnp
from jax.scipy.special import logsumexp, rel_entr

from . import gauss
from ..neural_tangents.extended import empirical_ntk


def kldiv_ub(q, p):
    """Compute the closed-form Gaussian KL divergence."""
    q_weight = nn.softmax(q['logit'])
    p_weight = nn.softmax(p['logit'])
    cat_kldiv = rel_entr(q_weight, p_weight).sum()
    gauss_kldiv = vmap(gauss.kldiv_cf)(q, p)
    return cat_kldiv + q_weight @ gauss_kldiv


def logpdf(value, logit, mean, var):
    """Compute the Gaussian log density."""
    return logsumexp(
        vmap(gauss.logpdf, in_axes=(None, 0, 0))(value, mean, var),
        b=nn.softmax(logit), axis=-1
    )


def kldiv_mc(sample, q, p):
    """Compute the Monte Carlo Gaussian KL divergence."""
    return vmap(
        lambda x:
        logpdf(x, q['logit'], q['mean'], q['var'])
        - logpdf(x, p['logit'], p['mean'], p['var'])
    )(sample).mean()


def get_param(params):
    """Compute the parameters of the parameters."""
    return {
        'logit': params['logit'],
        'mean': params['mean'],
        'var': tree_util.tree_map(lambda x: nn.softplus(x) ** 2, params['msd'])
    }


def get_output(params, apply, xs):
    """Compute the parameters of the outputs."""
    param = get_param(params)
    center = tree_util.tree_map(lambda x: x.mean(axis=0), param['mean'])
    mean = vmap(
        lambda m:
        apply({'params': center}, xs)
        + jvp(
            lambda var: apply(var, xs),
            ({'params': center},),
            ({'params': tree_util.tree_map(sub, m, center)},)
        )[1]
    )(param['mean'])
    var = vmap(
        empirical_ntk(apply, (), (0, 1), 0),
        in_axes=(None, None, None, 0)
    )(xs, None, {'params': center}, {'params': param['var']})
    return {'logit': params['logit'], 'mean': mean, 'var': var}


def sample(key, n, target, n_comp):
    """Generate Gaussian and Gumbel samples."""
    key1, key2 = random.split(key)
    return gauss.sample(
        key1, n, vmap(lambda x: target)(jnp.zeros(n_comp))
    ) | {'gumbel': random.gumbel(key2, shape=(n, n_comp))}


def transform(q, sample):
    """Transform a Gaussian mixture sample."""
    weight = nn.softmax(1000 * (q['logit'] + sample['gumbel']))
    gauss_sample = gauss.transform(q, sample)
    return vmap(
        lambda w, g: tree_util.tree_map(
            lambda x: jnp.tensordot(x, w, axes=(0, 0)), g
        )
    )(weight, gauss_sample)
