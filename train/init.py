"""Initialization functions."""

from jax import random, vmap
import jax.numpy as jnp


def standard_init(key, model, input_shape):
    """Initialize parameters for standard MLE/MAP training."""
    return model.init(key, jnp.zeros((1, *input_shape)))['params']


def gvi_init(key, model, input_shape):
    """Initialize parameters for Gaussian variational inference."""
    key1, key2 = random.split(key)
    return {
        'mean': standard_init(key1, model, input_shape),
        'msd': standard_init(key2, model, input_shape)
    }


def gmvi_init(key, n_comp, model, input_shape):
    """Initialize parameters for Gaussian-mixture variational inference."""
    key1, key2 = random.split(key)
    return {
        'logit': jnp.zeros((n_comp,)),
        'mean': vmap(standard_init, in_axes=(0, None, None))(
            random.split(key1, num=n_comp), model, input_shape
        ),
        'msd': vmap(standard_init, in_axes=(0, None, None))(
            random.split(key2, num=n_comp), model, input_shape
        ),
    }
