"""Models for Iris-SR."""

from flax import linen as nn
import jax.numpy as jnp
from jax import flatten_util, random, tree_util
import optax

from .. import TrainState


model = nn.Dense(3)
key = random.PRNGKey(1337)
key, key1 = random.split(key)
params = model.init(key, jnp.zeros(4))['params']
state_init = TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=optax.adam(0.01),
    hyperparams={}
)

class Consolidator(nn.Module):

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(50)(x)
        x = nn.swish(x)
        x = nn.Dense(1)(x)
        return x


consolidator = Consolidator()
key, key1, key2 = random.split(key, num=3)
pflat = flatten_util.ravel_pytree(params)[0]
state_consolidator_init = TrainState.create(
    apply_fn=consolidator.apply,
    params=consolidator.init(
        key1, jnp.expand_dims(pflat, 0)
    )['params'],
    tx=optax.adam(0.01),
    hyperparams={
        'minimum': tree_util.tree_map(jnp.zeros_like, state_init.params),
        'radius': 20.0,
        'ball': random.ball(key2, len(pflat), shape=(10000,))
    }
)