"""Training package."""

from operator import add

from flax.training import train_state
from jax import grad, jit, tree_util
import optax


class TrainState(train_state.TrainState):
    hyperparams: dict


def make_step(loss):
    return jit(
        lambda state, x, y: state.apply_gradients(
            grads=grad(loss)(state.params, x, y)
        )
    )


def make_loss_sse(state):
    return jit(
        lambda params, x, y:
        optax.l2_loss(state.apply_fn({'params': params}, x)[:, 0], y).sum()
    )


def make_loss_bce(state):
    return jit(
        lambda params, x, y:
        optax.sigmoid_binary_cross_entropy(
            state.apply_fn({'params': params}, x)[:, 0], y
        ).sum()
    )


def make_loss_sce(state):
    return jit(
        lambda params, x, y:
        optax.softmax_cross_entropy_with_integer_labels(
            state.apply_fn({'params': params}, x), y
        ).sum()
    )


def make_loss_reg(state, loss):
    return jit(
        lambda params, x, y: 
        tree_util.tree_reduce(
            add,
            tree_util.tree_map(
                lambda x: 0.5 * (state.hyperparams['precision'] * x ** 2).sum(),
                params
            )
        ) + loss(params, x, y)
    )
