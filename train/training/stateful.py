"""Stateful training."""

from jax import jit, random, value_and_grad


def make_step(root_dropout_key, loss):
    """Make a gradient-descent step function for a loss function."""

    @jit
    def step(state, var, xs, ys):
        dropout_key = random.fold_in(root_dropout_key, state.step)
        (loss_val, var), grads = value_and_grad(loss, has_aux=True)(
            state.params, var, {'dropout': dropout_key}, xs, ys
        )
        return state.apply_gradients(grads=grads), var

    return step
