"""Functions for the training state."""

from itertools import cycle

from jax import grad, jit, random, vmap
import jax.numpy as jnp

from dataops.array import batch, shuffle


def map_init(key, model, input_shape):
    """Initialize parameters for MLE/MAP training."""
    return model.init(key, jnp.zeros((1, *input_shape)))['params']


def gvi_init(key, model, input_shape):
    """Initialize parameters for Gaussian variational inference."""
    key1, key2 = random.split(key)
    return {
        'mean': map_init(key1, model, input_shape),
        'msd': map_init(key2, model, input_shape)
    }


def gmvi_init(key, n_comp, model, input_shape):
    """Initialize parameters for Gaussian-mixture variational inference."""
    key1, key2 = random.split(key)
    return {
        'logit': jnp.zeros((n_comp,)),
        'mean': vmap(map_init, in_axes=(0, None, None))(
            random.split(key1, num=n_comp), model, input_shape
        ),
        'msd': vmap(map_init, in_axes=(0, None, None))(
            random.split(key2, num=n_comp), model, input_shape
        ),
    }


def regular_sgd(key, n_epochs, batch_size, loss, state, xs, ys):
    """Generate epochly states for SGD."""
    step = make_step(loss)
    state1 = state
    for key1 in random.split(key, num=n_epochs):
        for indices in batch(batch_size, shuffle(key1, len(ys))):
            state1 = step(state1, xs[indices], ys[indices])
    return state1


def serial_sgd(
    key, n_epochs, batch_size, loss, state, xs, ys, coreset
):
    """Generate epochly states for SGD with coreset in series."""
    coreset.create_memmap()
    step = make_step(loss)
    state1 = state
    for key1 in random.split(key, num=n_epochs):
        key2, key3 = random.split(key1)
        for indices in batch(batch_size, shuffle(key2, len(ys))):
            state1 = step(state1, xs[indices], ys[indices])
        for xs_batch, ys_batch in coreset.shuffle_batch(key3):
            state1 = step(state1, xs_batch, ys_batch)
    coreset.delete_memmap()
    return state1


def parallel_sgd_shuffle_batch(
    key, n_epochs, batch_size, loss, state, xs, ys, coreset
):
    """Generate states for SGD with coreset in parallel by choice."""
    coreset.create_memmap()
    step = make_step(loss)
    state1 = state
    for key1 in random.split(key, num=n_epochs):
        key2, key3 = random.split(key1)
        for (indices, (xs_batch, ys_batch)) in zip(
            batch(batch_size, shuffle(key2, len(ys))),
            cycle(coreset.shuffle_batch(key3))
        ):
            state1 = step(
                state1, xs[indices], ys[indices], xs_batch, ys_batch
            )
    coreset.delete_memmap()
    return state1


def parallel_sgd_choice(
    key, n_epochs, batch_size, loss, state, xs, ys, coreset
):
    """Generate states for SGD with coreset in parallel by shuffling."""
    coreset.create_memmap()
    step = make_step(loss)
    state1 = state
    for key1 in random.split(key, num=n_epochs):
        keys = random.split(key1, num=-(len(ys) // -batch_size) + 1)
        for i, indices in enumerate(
            batch(batch_size, shuffle(keys[0], len(ys))), start=1
        ):
            state1 = step(
                state1, xs[indices], ys[indices], *coreset.choice(keys[i])
            )
    coreset.delete_memmap()
    return state1


def make_step(loss):
    """Make a gradient-descent step function for a loss function."""
    return jit(
        lambda state, *args: state.apply_gradients(
            grads=grad(loss)(state.params, *args)
        )
    )
