"""Trainer base class and mixins."""

from abc import ABC, abstractmethod

from flax.training.train_state import TrainState
from jax import random
from optax import adam

from dataops.array import get_pass_size

from ..predictor.stateless import (
    MAPPredictor, GaussPredictor, GSGaussPredictor, TPredictor
)
from ..training.stateless import init, gauss_init, gsgauss_init, t_init
from ..training.loss.stateless import get_nll
from ..training.vi import gauss, gaussmix, t


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

    def sample(self, key, xs=None):
        """Draw a standard sample for the parameter."""
        return gauss.sample(
            key, self.immutables['sample_size'],
            self.precomputed['param_example'] if xs is None
            else self.model.apply(
                {'params': self.precomputed['param_example']}, xs
            )
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

    def sample(self, key, xs=None):
        """Draw a standard sample for the parameter."""
        return gaussmix.sample(
            key, self.immutables['sample_size'],
            self.precomputed['param_example'] if xs is None
            else self.model.apply(
                {'params': self.precomputed['param_example']}, xs
            ),
            self.immutables['n_comp']
        )


class TMixin:
    """Mixin for t variation inference."""

    predictor_class = TPredictor

    def init_state(self):
        """Initialize the state."""
        return TrainState.create(
            apply_fn=self.model.apply,
            params=t_init(
                self.precomputed['keys']['init_state'],
                self.model,
                self.model_spec.in_shape
            ),
            tx=adam(self.immutables['lr'])
        )

    def sample(self, key, xs=None):
        """Draw a standard sample for the parameter."""
        return t.sample(
            key, self.immutables['sample_size'],
            self.precomputed['param_example'] if xs is None
            else self.model.apply(
                {'params': self.precomputed['param_example']}, xs
            ),
            self.immutables['df']
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
