"""Mixins for the training state."""

from flax.training.train_state import TrainState
from optax import adam

from .functions import init, gauss_init, gsgauss_init
from ..predict import MAPPredictor, GaussPredictor, GSGaussPredictor
from ..probability import gauss_sample, gsgauss_sample


class MAPMixin:
    """Mixin for MAP inference."""

    predictor = MAPPredictor

    def init_state(self):
        """Initialize the state."""
        return TrainState.create(
            apply_fn=self.model.apply,
            params=init(
                self.precomputed['keys']['init_state'],
                self.model, self.model_spec.in_shape
            ),
            tx=adam(self.immutables['lr'])
        )


class GaussMixin:
    """Mixin for Gaussian variation inference."""

    predictor = GaussPredictor

    def init_state(self):
        """Initialize the state."""
        return TrainState.create(
            apply_fn=self.model.apply,
            params=gauss_init(
                self.precomputed['keys']['init_state'],
                self.model,
                self.model_spec.in_shape
            ),
            tx=adam(self.immutables['lr'])
        )

    def sample(self, key):
        """Draw a standard sample for the reparameterization trick."""
        return gauss_sample(
            key, self.immutables['sample_size'],
            self.precomputed['param_example']
        )


class GSGaussMixin:
    """Mixin for Gaussian-mixture variation inference."""

    predictor = GSGaussPredictor

    def init_state(self):
        """Initialize the state."""
        return TrainState.create(
            apply_fn=self.model.apply,
            params=gsgauss_init(
                self.precomputed['keys']['init_state'],
                self.model,
                self.immutables['n_comp'],
                self.model_spec.in_shape
            ),
            tx=adam(self.immutables['lr'])
        )

    def sample(self, key):
        """Draw a standard sample for the reparameterization trick."""
        return gsgauss_sample(
            key, self.immutables['sample_size'],
            self.immutables['n_comp'], self.precomputed['param_example']
        )
