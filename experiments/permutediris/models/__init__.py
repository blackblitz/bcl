"""Models for Permuted Iris."""

import jax.numpy as jnp
from jax import flatten_util, random, tree_util
import optax

from train import TrainState


def make_state(main, consolidator):
    x = jnp.zeros((1, 4))
    keys = random.split(random.PRNGKey(1337), num=3)
    params_main = main.init(keys[0], x)['params']
    state_main = TrainState.create(
        apply_fn=main.apply,
        params=params_main,
        tx=optax.adam(0.01),
        hyperparams={}
    )
    pflat = flatten_util.ravel_pytree(params_main)[0]
    params_consolidator = consolidator.init(
        keys[1], jnp.expand_dims(pflat, 0)
    )['params']
    state_consolidator = TrainState.create(
        apply_fn=consolidator.apply,
        params=params_consolidator,
        tx=optax.adam(0.01),
        hyperparams={
            'minimum': tree_util.tree_map(jnp.zeros_like, params_main),
            'radius': 20.0,
            'ball': random.ball(keys[2], len(pflat), shape=(10000,))
        }
    )
    return state_main, state_consolidator
