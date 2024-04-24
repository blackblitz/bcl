"""Models for pre-training for Split MNIST."""

from flax.training.train_state import TrainState
import jax.numpy as jnp
from jax import random
import optax


def make_state_pretrained(model):
    x = jnp.zeros((1, 28, 28, 1))
    keys = random.split(random.PRNGKey(1337), num=3)
    params = model.init(keys[0], x)['params']
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.adam(0.001)
    )
    return state
