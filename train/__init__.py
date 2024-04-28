"""Training package."""

from jax import grad, jit, random, tree_util
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


def normaltree(key, n, tree):
    keys = random.split(key, len(tree_util.tree_leaves(tree)))
    keys = tree_util.tree_unflatten(tree_util.tree_structure(tree), keys)
    return tree_util.tree_map(
        lambda x, key: random.normal(key, (n, *x.shape)), tree, keys
    )
