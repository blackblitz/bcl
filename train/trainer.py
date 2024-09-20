"""Base classes."""

from abc import ABC, abstractmethod
from enum import Enum, unique

from flax.training.train_state import TrainState
from jax import jit, random
import jax.numpy as jnp
from optax import sgd

from dataops.array import get_pass_size

from .state.functions import init


@unique
class NNType(Enum):
    """Neural-network type."""

    SIGMOID = 1
    SOFTMAX = 2


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
            'pass_size': get_pass_size(self.model_spec.in_shape),
            'param_example': init(
                random.PRNGKey(1337), self.model, self.model_spec.in_shape
            )
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
