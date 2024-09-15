"""Functions related to probability."""

import math
from operator import sub

import distrax
from jax import jvp, random, tree_util, vmap
from jax.nn import softmax, softplus
import jax.numpy as jnp
from jax.scipy.special import logsumexp, rel_entr
from toolz import compose

from dataops import tree

from .neural_tangents.extended import empirical_ntk


def get_mean_var(params):
    """Compute the means and variances from a params."""
    return (
        params['mean'],
        tree_util.tree_map(lambda x: softplus(x) ** 2, params['msd'])
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


def gauss_kldiv(mean1, var1, mean2, var2):
    """Compute the KL divergence of Gaussian PDFs."""
    return distrax.Normal(mean1, jnp.sqrt(var1)).kl_divergence(
        distrax.Normal(mean2, jnp.sqrt(var2))
    )


def gauss_params_kldiv(params1, params2):
    """Compute the KL divergence of diagonal Gaussian parameters."""
    return tree.sum(
        tree_util.tree_map(
            gauss_kldiv,
            *get_mean_var(params1),
            *get_mean_var(params2)
        )
    )


def gauss_logpdf(x, mean, var):
    """Compute the Gaussian log density."""
    return distrax.Normal(mean, jnp.sqrt(var)).log_prob(x)


def gauss_params_logpdf(x, params):
    """Compute the log density of diagonal Gaussian parameters."""
    return tree.sum(
        tree_util.tree_map(gauss_logpdf, x, *get_mean_var(params))
    )


def gaussmix_params_logpdf(x, params):
    """Compute the log density of GM parameters."""
    return logsumexp(
        vmap(gauss_params_logpdf, in_axes=(None, 0))(x, params),
        b=softmax(params['logit']), axis=-1
    )


def gaussmix_params_kldiv_mc(param_sample, params1, params2):
    """Compute the KL divergence of GM parameters by MC integration."""
    return vmap(
        lambda x: gaussmix_params_logpdf(x, params1)
        - gaussmix_params_logpdf(x, params2)
    )(param_sample).mean()


def gaussmix_params_kldiv_ub(params1, params2):
    """Compute an upper-bound KL divergence of diagonal GM parameters."""
    weight1 = softmax(params1['logit'])
    weight2 = softmax(params2['logit'])
    cat_kldiv_val = rel_entr(weight1, weight2).sum()
    gauss_kldiv_val = vmap(gauss_params_kldiv)(params1, params2)
    return cat_kldiv_val + weight1 @ gauss_kldiv_val


def gaussmix_kldiv_mc(key, n, weight1, mean1, var1, weight2, mean2, var2):
    """Compute the KL divergence of Gaussian-mixture PDFs."""
    f = vmap(jnp.ravel)
    dist1 = distrax.MixtureSameFamily(
        distrax.Categorical(weight1),
        distrax.MultivariateNormalDiag(f(mean1), f(jnp.sqrt(var1)))
    )
    dist2 = distrax.MixtureSameFamily(
        distrax.Categorical(weight2),
        distrax.MultivariateNormalDiag(f(mean2), f(jnp.sqrt(var2)))
    )
    sample, logpdf1 = dist1.sample_and_log_prob(seed=key, sample_shape=(n,))
    logpdf2 = dist2.log_prob(sample)
    return (logpdf1 - logpdf2).mean()


def gauss_output_kldiv(params, prior, apply, xs):
    """Compute the KL divergence of Gaussian outputs."""
    mean, var = get_mean_var(params)
    prior_mean, prior_var = get_mean_var(prior)
    output_mean = apply({'params': mean}, xs)
    prior_output_mean = apply({'params': prior_mean}, xs)
    get_output_var = empirical_ntk(apply, (), (0, 1), 0)
    output_var = get_output_var(
        xs, None, {'params': mean}, {'params': var}
    )
    prior_output_var = get_output_var(
        xs, None, {'params': prior_mean}, {'params': prior_var}
    )
    return gauss_kldiv(
        output_mean, output_var, prior_output_mean, prior_output_var
    ).sum()


def gaussmix_output_params(params, prior, apply, xs):
    """Compute the parameters of the output distribution."""
    mean, var = get_mean_var(params)
    prior_mean, prior_var = get_mean_var(prior)
    center = tree_util.tree_map(lambda x: x.mean(axis=0), mean)
    prior_center = tree_util.tree_map(lambda x: x.mean(axis=0), prior_mean)

    def get_output_mean(center, mean):
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

    output_mean = get_output_mean(center, mean)
    prior_output_mean = get_output_mean(prior_center, prior_mean)
    get_output_var = empirical_ntk(apply, (), (0, 1), 0)
    output_var = vmap(get_output_var, in_axes=(None, None, None, 0))(
        xs, None, {'params': center}, {'params': var}
    )
    prior_output_var = vmap(get_output_var, in_axes=(None, None, None, 0))(
        xs, None, {'params': prior_center}, {'params': prior_var}
    )
    weight = softmax(params['logit'])
    prior_weight = softmax(prior['logit'])
    return (
        weight, output_mean, output_var,
        prior_weight, prior_output_mean, prior_output_var
    )


def gaussmix_output_kldiv_mc(key, n, params, prior, apply, xs):
    """Compute the KL divergence of GM outputs by MC integration."""
    (
        weight, output_mean, output_var,
        prior_weight, prior_output_mean, prior_output_var
    ) = gaussmix_output_params(params, prior, apply, xs)
    return vmap(
        vmap(gaussmix_kldiv_mc, in_axes=(None, None, None, 1, 1, None, 1, 1)),
        in_axes=(None, None, None, 2, 2, None, 2, 2)
    )(
        key, n, weight, output_mean, output_var,
        prior_weight, prior_output_mean, prior_output_var
    ).sum()


def gaussmix_output_kldiv_ub(params, prior, apply, xs):
    """Compute an upper-bound KL divergence of Gaussian-mixture outputs."""
    (
        weight, output_mean, output_var,
        prior_weight, prior_output_mean, prior_output_var
    ) = gaussmix_output_params(params, prior, apply, xs)
    cat_kldiv_val = rel_entr(weight, prior_weight).sum()
    gauss_kldiv_val = vmap(compose(jnp.sum, gauss_kldiv))(
        output_mean, output_var, prior_output_mean, prior_output_var
    )
    return cat_kldiv_val + weight @ gauss_kldiv_val


def gauss_sample(key, n, target):
    """Generate a Gaussian sample."""
    keys = random.split(key, len(tree_util.tree_leaves(target)))
    keys = tree_util.tree_unflatten(tree_util.tree_structure(target), keys)
    return {
        'gauss': tree_util.tree_map(
            lambda x, key: random.normal(key, (n, *x.shape)), target, keys
        )
    }


def gsgauss_sample(key, n, m, target):
    """Generate Gaussian and Gumbel samples."""
    key1, key2 = random.split(key)
    return gauss_sample(
        key1, n, tree_util.tree_map(
            lambda x: jnp.repeat(jnp.expand_dims(x, 0), m, axis=0), target
        )
    ) | {'gumbel': random.gumbel(key2, (n, m))}


def gauss_param(params, sample):
    """Parameterize a Gaussian sample."""
    return tree_util.tree_map(
        lambda m, r, zs: m + softplus(r) * zs,
        params['mean'], params['msd'], sample['gauss']
    )


def gsgauss_param(params, sample):
    """Parameterize a Gumbel-softmax-Gaussian-mixture sample."""
    weight = softmax(1000 * (params['logit'] + sample['gumbel']))
    gauss = gauss_param(params, sample)
    return vmap(
        lambda w, g: tree_util.tree_map(
            lambda x: jnp.tensordot(x, w, axes=(0, 0)), g
        )
    )(weight, gauss)
