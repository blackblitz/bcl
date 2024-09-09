"""Base classes."""

from abc import ABC, abstractmethod
from itertools import chain
import math
from pathlib import Path

from flax.training.train_state import TrainState
from jax import grad, jit, random
from optax import adam, sgd

from dataops.array import draw_batches
from dataops.io import zarr_to_memmap

from .loss import sigmoid_ce, softmax_ce
from .predict import sigmoid_map, softmax_map
from .init import standard_init


def make_step(loss):
    """Make a gradient-descent step function for a loss function."""
    return jit(
        lambda state, *args: state.apply_gradients(
            grads=grad(loss)(state.params, *args)
        )
    )


class MAPMixin:
    """Mixin for MAP inference."""

    def init_state(self):
        """Initialize the state."""
        return TrainState.create(
            apply_fn=self.model.apply,
            params=standard_init(
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
            'pass_size': 2 ** math.floor(
                20 * math.log2(2)
                - 2 - sum(map(math.log2, self.metadata['input_shape']))
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
        self.update_state(xs, ys)
        self.update_mutables(xs, ys)

    @abstractmethod
    def make_predict(self):
        """Make a predicting function."""


class StandardTrainer(ContinualTrainer):
    """Standard continual trainer which uses mini-batch gradient descent."""

    def update_state(self, xs, ys):
        """Update the training state."""
        step = make_step(self.loss)
        for key in random.split(
            self.precomputed['keys']['update_state'],
            num=self.immutables['n_epochs']
        ):
            for xs_batch, ys_batch in draw_batches(
                key, self.immutables['batch_size'], xs, ys
            ):
                self.state = step(self.state, xs_batch, ys_batch)


class SerialTrainer(ContinualTrainer):
    """Continual trainer which uses a coreset in series."""

    def update_state(self, xs, ys):
        """Update the training state."""
        xs_path = Path('coreset_xs.npy')
        ys_path = Path('coreset_ys.npy')
        coreset_xs, coreset_ys = zarr_to_memmap(
            self.mutables['coreset'], xs_path, ys_path
        )
        step = make_step(self.loss)
        for key in random.split(
            self.precomputed['keys']['update_state'],
            num=self.immutables['n_epochs']
        ):
            for xs_batch, ys_batch in chain(
                draw_batches(key, self.immutables['batch_size'], xs, ys),
                draw_batches(
                    key, self.immutables['batch_size'],
                    coreset_xs, coreset_ys
                )
            ):
                self.state = step(self.state, xs_batch, ys_batch)
        xs_path.unlink()
        ys_path.unlink()


class ParallelTrainer(ContinualTrainer):
    """Continual trainer which uses a coreset in parallel."""

    def update_state(self, xs, ys):
        """Update the training state."""
        xs_path = Path('coreset_xs.npy')
        ys_path = Path('coreset_ys.npy')
        coreset_xs, coreset_ys = zarr_to_memmap(
            self.mutables['coreset'], xs_path, ys_path
        )
        step = make_step(self.loss)
        for key in random.split(
            self.precomputed['keys']['update_state'],
            num=self.immutables['n_epochs']
        ):
            if len(coreset_ys) == 0:
                for xs_batch, ys_batch in draw_batches(
                    key, self.immutables['batch_size'], xs, ys
                ):
                    self.state = step(
                        self.state,
                        xs_batch, ys_batch,
                        coreset_xs[[]], coreset_ys[[]]
                    )
            else:
                n_batches = -(len(ys) // -self.immutables['batch_size'])
                keys = random.split(key, num=n_batches + 1)
                for i, (xs_batch, ys_batch) in enumerate(draw_batches(
                    keys[0], self.immutables['batch_size'], xs, ys
                ), start=1):
                    indices = random.choice(
                        keys[i], len(coreset_ys),
                        shape=(self.immutables['batch_size'],),
                        replace=False
                    )
                    self.state = step(
                        self.state,
                        xs_batch, ys_batch,
                        coreset_xs[indices], coreset_ys[indices]
                    )
        xs_path.unlink()
        ys_path.unlink()


class Finetuning(MAPMixin, StandardTrainer):
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

    def update_mutables(self, xs, ys):
        """Update the mutable hyperparameters."""
