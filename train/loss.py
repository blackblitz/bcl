"""Loss functions."""

from operator import add

from jax import grad, jit, random, tree_util


def make_loss_single_output(state, loss_point):
    return jit(
        lambda params, x, y: loss_point(
            state.apply_fn({'params': params}, x)[:, 0], y
        ).sum()
    )


def make_loss_multi_output(state, loss_point):
    return jit(
        lambda params, x, y: loss_point(
            state.apply_fn({'params': params}, x), y
        ).sum()
    )


def make_loss_reg(state, precision, loss_basic):
    return jit(
        lambda params, x, y: 
        tree_util.tree_reduce(
            add,
            tree_util.tree_map(
                lambda x: 0.5 * (precision * x ** 2).sum(), params
            )
        ) + loss_basic(params, x, y)
    )


def make_step(loss):
    return jit(
        lambda state, x, y: state.apply_gradients(
            grads=grad(loss)(state.params, x, y)
        )
    )
