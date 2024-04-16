"""Models for Split MNIST."""

from flax.training.train_state import TrainState
import jax.numpy as jnp
from jax import flatten_util, random, tree_util
import optax


def make_state(main, consolidator):
    x = jnp.zeros((1, 28, 28, 1))
    keys = random.split(random.PRNGKey(1337), num=3)
    params_main = main.init(keys[0], x)['params']
    state_main = TrainState.create(
        apply_fn=main.apply,
        params=params_main,
        tx=optax.adam(0.01)
    )
    pflat = flatten_util.ravel_pytree(params_main)[0]
    params_consolidator = consolidator.init(
        keys[1], jnp.expand_dims(pflat, 0)
    )['params']
    state_consolidator = TrainState.create(
        apply_fn=consolidator.apply,
        params=params_consolidator,
        tx=optax.adam(0.01)
    )
    return state_main, state_consolidator
