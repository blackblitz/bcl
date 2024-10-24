"""Functions related to probability."""

import math
from operator import sub

import distrax
from jax import jvp, random, tree_util, vmap
from jax.nn import softmax, softplus
import jax.numpy as jnp
from jax.scipy import stats
from jax.scipy.special import entr, gammaln, logsumexp, rel_entr
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


def get_t_prior(invscale, params):
    """Return the t prior parameters with fixed inverse scale."""
    rs = math.sqrt(1 / invscale)
    mrs = rs + math.log(-math.expm1(-rs))
    return {
        'loc': tree_util.tree_map(jnp.zeros_like, params['loc']),
        'ms': tree_util.tree_map(
            lambda x: jnp.full_like(x, mrs), params['ms']
        )
    }


def gauss_kldiv(mean1, var1, mean2, var2):
    """Compute the KL divergence of Gaussian PDFs."""
    var_ratio = var1 / var2
    return 0.5 * (
        -jnp.log(var_ratio) + var_ratio + (mean1 - mean2) ** 2 / var2 - 1
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


def gauss_logpdf(xs, mean, var):
    """Compute the Gaussian log density."""
    return vmap(jnp.sum)(stats.norm.logpdf(xs, loc=mean, scale=jnp.sqrt(var)))


def gauss_params_logpdf(x, params):
    """Compute the log density of diagonal Gaussian parameters."""
    return tree.sum(
        tree_util.tree_map(gauss_logpdf, x, *get_mean_var(params))
    )


def gaussmix_logpdf(x, weight, mean, var):
    """Compute the Gaussian mixture log density."""
    return logsumexp(
        gauss_logpdf(x, mean, var),
        #vmap(gauss_logpdf, in_axes=(None, 0, 0))(x, mean, var),
        b=weight, axis=-1
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
    sample = dist1.sample(
        seed=key, sample_shape=(n,)
    )#.reshape((-1, *mean1.shape))
    logpdf1 = gaussmix_logpdf(sample, weight1, f(mean1), f(var1))
    logpdf2 = gaussmix_logpdf(sample, weight2, f(mean2), f(var2))
    #logpdf1 = dist1.log_prob(sample)
    #logpdf2 = dist2.log_prob(sample)
    return (logpdf1 - logpdf2).mean()


def t_kldiv_mc(key, n, loc1, scale1, loc2, scale2, df):
    """Compute the KL divergence of t PDFs."""
    key1, key2 = random.split(key)
    gauss = random.normal(key1, shape=(n, *loc1.shape))
    gamma = 2 / df * random.gamma(key2, df / 2, shape=(n,))
    sample = vmap(lambda x, y: x / y)(
        loc1 + jnp.sqrt(scale1) * gauss,
        jnp.sqrt(gamma)
    )
    logpdf1 = t_logpdf(sample, loc1, scale1, df)
    logpdf2 = t_logpdf(sample, loc2, scale2, df)
    return (logpdf1 - logpdf2).mean()


def get_loc_scale(params):
    """Compute the means and variances from a params."""
    return (
        params['loc'],
        tree_util.tree_map(lambda x: softplus(x) ** 2, params['mrs'])
    )


def t_logpdf(x, loc, scale, df):
    """Compute the t log density."""
    size = x.size
    return (
        gammaln(0.5 * (df + size)) - gammaln(0.5 * df)
        - 0.5 * size * jnp.log(df) - 0.5 * size * jnp.log(jnp.pi)
        - 0.5 * jnp.log(scale).sum()
        - 0.5 * (df + size) * jnp.log(1 + ((x - loc) ** 2 / scale).sum() / df)
    )


def t_params_logpdf(x, params, df):
    """Compute the log density of t parameters."""
    size = tree.size(x)
    loc, scale = get_loc_scale(params)
    return (
        gammaln(0.5 * (df + size)) - gammaln(0.5 * df)
        - 0.5 * size * jnp.log(df) - 0.5 * size * jnp.log(jnp.pi)
        - 0.5 * tree.sum(tree_util.tree_map(jnp.log, scale))
        - 0.5 * (df + size) * jnp.log(
            1 + tree.sum(
                tree_util.tree_map(
                    lambda xl, locl, scalel: (xl - locl) ** 2 / scalel,
                    x, loc, scale
                )
            ) / df
        )
    )


def t_params_kldiv_mc(param_sample, params1, params2, df):
    """Compute the KL divergence of PDFs over t parameters."""
    return vmap(
        lambda x: t_params_logpdf(x, params1, df)
        - t_params_logpdf(x, params2, df)
    )(param_sample).mean()


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
    return gaussmix_kldiv_mc(
        key, n, weight, output_mean, output_var,
        prior_weight, prior_output_mean, prior_output_var
    )


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


def t_output_kldiv_mc(key, n, params, prior, df, apply, xs):
    """Compute the KL divergence of t outputs."""
    loc, scale = get_loc_scale(params)
    prior_loc, prior_scale = get_loc_scale(prior)
    output_loc = apply({'params': loc}, xs)
    prior_output_loc = apply({'params': prior_loc}, xs)
    get_output_scale = empirical_ntk(apply, (), (0, 1), 0)
    output_scale = get_output_scale(
        xs, None, {'params': loc}, {'params': scale}
    )
    prior_output_scale = get_output_scale(
        xs, None, {'params': prior_loc}, {'params': prior_scale}
    )
    return t_kldiv_mc(
        key, n,
        output_loc, output_scale,
        prior_output_loc, prior_output_scale,
        df
    )


def gauss_sample(key, n, target):
    """Generate a Gaussian sample."""
    return {
        'gauss': vmap(
            lambda x: tree.gauss(x, target)
        )(random.split(key, num=n))
    }


def gsgauss_sample(key, n, m, target):
    """Generate Gaussian and Gumbel samples."""
    key1, key2 = random.split(key)
    return gauss_sample(
        key1, n, vmap(lambda x: target)(jnp.zeros(m))
    ) | {'gumbel': random.gumbel(key2, shape=(n, m))}


def t_sample(key, n, df, target):
    """Generate Gaussian and chi-square samples."""
    key1, key2 = random.split(key)
    return {
        'gauss': vmap(
            lambda x: tree.gauss(x, target)
        )(random.split(key1, num=n)),
        'gamma': 2 / df * random.gamma(key2, df / 2, shape=(n,))
    }


def gauss_transform(params, sample):
    """Transform a Gaussian sample."""
    return vmap(
        lambda zs: tree_util.tree_map(
            lambda m, r, z: m + softplus(r) * z,
            params['mean'], params['msd'], zs
        )
    )(sample['gauss'])


def gsgauss_transform(params, sample):
    """Transform a Gumbel-softmax-Gaussian-mixture sample."""
    weight = softmax(1000 * (params['logit'] + sample['gumbel']))
    gauss = gauss_param(params, sample)
    return vmap(
        lambda w, g: tree_util.tree_map(
            lambda x: jnp.tensordot(x, w, axes=(0, 0)), g
        )
    )(weight, gauss)


def t_transform(params, sample):
    """Transform a t sample."""
    return vmap(
        lambda zs, u: tree_util.tree_map(
            lambda m, r, z: m + softplus(r) / jnp.sqrt(u) * z,
            params['loc'], params['mrs'], zs
        )
    )(sample['gauss'], sample['gamma'])


def bern_entr(p):
    """Calculate the entropy of Bernoulli random variables."""
    return entr(p) + entr(1 - p)


def cat_entr(p):
    """Calculate the entropy of categorical random variables."""
    return entr(p).sum(axis=-1)
