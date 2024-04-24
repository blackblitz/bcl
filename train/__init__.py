"""Training package."""

from jax import grad, jit
import optax


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


def make_step(loss):
    return jit(
        lambda state, x, y: state.apply_gradients(
            grads=grad(loss)(state.params, x, y)
        )
    )
