"""Trainer base class and mixins."""

from abc import ABC, abstractmethod

from flax.training.train_state import TrainState
from jax import random
from optax import adam

from dataops.array import get_pass_size

from .loss.stateless import get_nll
from .predictor.stateless import MAPPredictor, GaussPredictor, GSGaussPredictor
from .probability import gauss_sample, gsgauss_sample
from .training.init import init, gauss_init, gsgauss_init


class MAPMixin:
    """Mixin for MAP inference."""

    predictor_class = MAPPredictor

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

    predictor_class = GaussPredictor

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

    predictor_class = GSGaussPredictor

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


class ContinualTrainer(ABC):
    """Abstract base class for continual learning."""

    def __init__(self, model, model_spec, immutables):
        """Intialize self."""
        self.model = model
        self.model_spec = model_spec
        self.immutables = immutables
        self.precomputed = self.precompute()
        self.state = self.init_state()
        self.mutables = self.init_mutables()
        self.loss = None

    def precompute(self):
        """Precompute."""
        key_names = [
            'init_state', 'init_mutables',
            'update_loss', 'update_state', 'update_mutables'
        ]
        return {
            'keys': dict(zip(
                key_names,
                random.split(
                    random.key(self.immutables['seed']),
                    num=len(key_names)
                )
            )),
            'pass_size': get_pass_size(self.model_spec.in_shape),
            'param_example': init(
                random.key(1337), self.model, self.model_spec.in_shape
            ),
            'nll': get_nll(self.model_spec.nll)(self.model.apply)
        }

    @abstractmethod
    def init_state(self):
        """Initialize the state."""

    @abstractmethod
    def init_mutables(self):
        """Initialize the mutable hyperparameters."""

    @abstractmethod
    def update_loss(self, xs, ys):
        """Update the loss function."""

    @abstractmethod
    def update_state(self, xs, ys):
        """Update the training state."""

    @abstractmethod
    def update_mutables(self, xs, ys):
        """Update the dynamic hyperparameters."""

    def train(self, xs, ys):
        """Train with a dataset."""
        self.update_loss(xs, ys)
        yield from self.update_state(xs, ys)
        self.update_mutables(xs, ys)
