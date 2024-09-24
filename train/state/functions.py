"""Functions for the training state."""

import jax.numpy as jnp
from jax import grad, jit, random, vmap

from dataops import tree


def init(key, model, input_shape):
    """Initialize parameters."""
    return model.init(key, jnp.zeros((1, *input_shape)))['params']


def gauss_init(key, model, input_shape):
    """Initialize parameters for Gaussian variational inference."""
    key1, key2 = random.split(key)
    mean = init(key1, model, input_shape)
    return {
        'mean': mean,
        'msd': tree.gauss(key2, mean, loc=-2.0, scale=0.05)
    }


def gsgauss_init(key, model, n_comp, input_shape):
    """Initialize parameters for Gaussian-mixture variational inference."""
    key1, key2, key3 = random.split(key, num=3)
    mean = vmap(init, in_axes=(0, None, None))(
        random.split(key1, num=n_comp), model, input_shape
    )
    return {
        'logit': random.normal(key2, (n_comp,)),
        'mean': mean,
        'msd': tree.gauss(key2, mean, loc=-2.0, scale=0.05)
    }


def make_step(loss):
    """Make a gradient-descent step function for a loss function."""
    return jit(
        lambda state, *args: state.apply_gradients(
            grads=grad(loss)(state.params, *args)
        )
    )
