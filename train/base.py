"""Base classes."""

from abc import ABC, abstractmethod

from flax.training.train_state import TrainState
from jax import jit, random
from optax import sgd

from dataops.io import get_pass_size

from .loss import sigmoid_ce, softmax_ce
from .predict import sigmoid_map, softmax_map
from .state import map_init, regular_sgd


class ContinualTrainer(ABC):
    """Abstract base class for continual learning."""

    def __init__(self, model, immutables, metadata):
        """Intialize self."""
        self.model = model
        self.immutables = immutables
        self.metadata = metadata
        self.precomputed = self.precompute()
        self.state = self.init_state()
        self.mutables = self.init_mutables()
        self.loss = None

    def _choose(self, option1, option2):
        """Choose an option based on binary or multi-class classification."""
        if len(self.metadata['classes']) == 2:
            return option1
        elif len(self.metadata['classes']) > 2:
            return option2
        else:
            raise ValueError('number of classes must be at least 2')

    def _make_keys(self, names):
        """Make keys for pseudo-random number generation."""
        return {
            'keys': dict(zip(
                names,
                random.split(random.PRNGKey(
                    self.immutables['seed']), num=len(names)
                )
            ))
        }

    def precompute(self):
        """Precompute."""
        return {
            'pass_size': get_pass_size(self.metadata['input_shape'])
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
        self.update_state(xs, ys)
        self.update_mutables(xs, ys)

    @abstractmethod
    def make_predict(self):
        """Make a predicting function."""


class MAPMixin:
    """Mixin for MAP inference."""

    def init_state(self):
        """Initialize the state."""
        return TrainState.create(
            apply_fn=self.model.apply,
            params=map_init(
                self.precomputed['keys']['init_state'],
                self.model, self.metadata['input_shape']
            ),
            tx=sgd(self.immutables['lr'])
        )

    def make_predict(self):
        """Make a predicting function."""
        return self._choose(sigmoid_map, softmax_map)(
            self.model.apply, self.state.params
        )


class Finetuning(MAPMixin, ContinualTrainer):
    """Fine-tuning for continual learning."""

    def precompute(self):
        """Precompute."""
        return super().precompute() | self._make_keys(
            ['init_state', 'update_state']
        )

    def init_mutables(self):
        """Initialize the mutable hyperparameters."""
        return {}

    def update_loss(self, xs, ys):
        """Update the loss function."""
        self.loss = jit(
            self._choose(sigmoid_ce, softmax_ce)(
                self.immutables['precision'], self.model.apply
            )
        )

    def update_state(self, xs, ys):
        """Update the training state."""
        self.state = regular_sgd(
            self.precomputed['keys']['update_state'],
            self.immutables['n_epochs'],
            self.immutables['batch_size'],
            self.loss,
            self.state,
            xs, ys
        )

    def update_mutables(self, xs, ys):
        """Update the mutable hyperparameters."""
