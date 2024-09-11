"""Mixins for initializing and updating the training state."""

from itertools import cycle

from flax.training.train_state import TrainState
from jax import grad, jit, random, vmap
import jax.numpy as jnp
from optax import sgd

from dataops.array import batch, shuffle

from .predict import MAPPredictor


def init(key, model, input_shape):
    """Initialize parameters."""
    return model.init(key, jnp.zeros((1, *input_shape)))['params']


class MAPMixin:
    """Mixin for MAP inference."""

    predictor = MAPPredictor

    def init_state(self):
        """Initialize the state."""
        return TrainState.create(
            apply_fn=self.model.apply,
            params=init(
                self.precomputed['keys']['init_state'],
                self.model, self.metadata['input_shape']
            ),
            tx=sgd(self.immutables['lr'])
        )


def gauss_init(key, model, input_shape):
    """Initialize parameters for Gaussian variational inference."""
    key1, key2 = random.split(key)
    return {
        'mean': init(key1, model, input_shape),
        'msd': init(key2, model, input_shape)
    }


class GaussMixin:
    """Mixin for Gaussian variation inference."""

    def init_state(self):
        """Initialize the state."""
        return TrainState.create(
            apply_fn=self.model.apply,
            params=gauss_init(
                self.precomputed['keys']['init_state'],
                self.model, self.metadata['input_shape']
            ),
            tx=sgd(self.immutables['lr'])
        )


def gsgauss_init(key, model, input_shape):
    """Initialize parameters for Gaussian-mixture variational inference."""
    key1, key2 = random.split(key)
    return {
        'logit': jnp.zeros((n_comp,)),
        'mean': vmap(init, in_axes=(0, None, None))(
            random.split(key1, num=n_comp), model, input_shape
        ),
        'msd': vmap(init, in_axes=(0, None, None))(
            random.split(key1, num=n_comp), model, input_shape
        )
    }

class GSGaussMixin:
    """Mixin for Gaussian-mixture variation inference."""

    def init_state(self):
        """Initialize the state."""
        return TrainState.create(
            apply_fn=self.model.apply,
            params=gsgauss_init(
                self.precomputed['keys']['init_state'],
                self.model, self.metadata['input_shape']
            ),
            tx=sgd(self.immutables['lr'])
        )


def make_step(loss):
    """Make a gradient-descent step function for a loss function."""
    return jit(
        lambda state, *args: state.apply_gradients(
            grads=grad(loss)(state.params, *args)
        )
    )


