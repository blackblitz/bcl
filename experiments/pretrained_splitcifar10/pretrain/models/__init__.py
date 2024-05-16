"""Models for pre-training for Split MNIST."""

from flax.training.train_state import TrainState
import jax.numpy as jnp
from jax import random
import optax


def make_state_pretrained(model):
    """Make state for the pre-trained model."""
    x = jnp.zeros((1, 32, 32, 3))
    params = model.init(random.PRNGKey(1337), x)['params']
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.adam(0.001)
    )
    return state
