"""Foundation model for pre-training for MNIST."""

from flax import linen as nn
import jax.numpy as jnp
from jax import random
import optax

from .. import TrainState


class CNN(nn.Module):
    def setup(self):
        self.feature_extractor = FeatureExtractor()
        self.classifier = nn.Dense(10)

    def __call__(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x


class FeatureExtractor(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(32, (4, 4))(x)
        x = nn.swish(x)
        x = nn.max_pool(x, (8, 8), strides=(8, 8))
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(8)(x)
        x = nn.swish(x)
        return x


model = CNN()
key = random.PRNGKey(1337)
key, key1 = random.split(key)
params = model.init(key, jnp.zeros((1, 28, 28, 1)))['params']
state_init = TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=optax.adam(0.01),
    hyperparams={}
)
