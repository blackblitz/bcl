"""Functions for the training state."""

import jax.numpy as jnp
from jax import grad, jit, random, vmap


def init(key, model, input_shape):
    """Initialize parameters."""
    return model.init(key, jnp.zeros((1, *input_shape)))['params']


def gauss_init(key, model, input_shape):
    """Initialize parameters for Gaussian variational inference."""
    key1, key2 = random.split(key)
    return {
        'mean': init(key1, model, input_shape),
        'msd': init(key2, model, input_shape)
    }


def gsgauss_init(key, model, n_comp, input_shape):
    """Initialize parameters for Gaussian-mixture variational inference."""
    key1, key2 = random.split(key)
    return {
        'logit': jnp.zeros((n_comp,)),
        'mean': vmap(init, in_axes=(0, None, None))(
            random.split(key1, num=n_comp), model, input_shape
        ),
        'msd': vmap(init, in_axes=(0, None, None))(
            random.split(key1, num=n_comp), model, input_shape
        )
    }


def make_step(loss):
    """Make a gradient-descent step function for a loss function."""
    return jit(
        lambda state, *args: state.apply_gradients(
            grads=grad(loss)(state.params, *args)
        )
    )
