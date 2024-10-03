"""Stateless training."""

from jax import grad, jit


def make_step(loss):
    """Make a gradient-descent step function for a loss function."""
    return jit(
        lambda state, *args: state.apply_gradients(
            grads=grad(loss)(state.params, *args)
        )
    )
