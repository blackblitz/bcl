"""Base classes."""

from abc import ABC, abstractmethod
from itertools import chain
import math
from pathlib import Path

from flax.training.train_state import TrainState
from jax import grad, jit, random
from optax import adam

from dataops.array import draw_batches
from dataops.io import zarr_to_memmap

from .loss import sigmoid_ce, softmax_ce
from .predict import sigmoid_map, softmax_map
from .init import standard_init


def get_pass_size(input_shape):
    """Calculate the batch size for passing through a dataset."""
    return 2 ** math.floor(
        20 * math.log2(2) - 2 - sum(map(math.log2, input_shape))
    )


def make_step(loss):
    """Make a gradient-descent step function for a loss function."""
    return jit(
        lambda state, *args: state.apply_gradients(
            grads=grad(loss)(state.params, *args)
        )
    )


class ContinualTrainer(ABC):
    """Abstract base class for continual learning."""

    def __init__(self, model, immutables, metadata):
        """Intialize self."""
        self.model = model
        self.immutables = immutables
        self.metadata = metadata
        self.state = self.init_state()
        self.mutables = self.init_mutables()

    def _choose(self, option1, option2):
        """Choose an option based on binary or multi-class classification."""
        if len(self.metadata['classes']) == 2:
            return option1
        elif len(self.metadata['classes']) > 2:
            return option2
        else:
            raise ValueError('number of classes must be at least 2')

    def init_state(self):
        """Initialize the state."""
        return TrainState.create(
            apply_fn=self.model.apply,
            params=standard_init(
                random.PRNGKey(self.immutables['init_state_seed']),
                self.model, self.metadata['input_shape']
            ),
            tx=adam(self.immutables['lr'])
        )

    def init_mutables(self):
        """Initialize the mutable hyperparameters."""
        return {
            'loss': self._choose(sigmoid_ce, softmax_ce)(
                self.immutables['precision'], self.model.apply
            ),
            'pass_size': get_pass_size(self.metadata['input_shape'])
        }

    @abstractmethod
    def update_state(self, xs, ys):
        """Update the training state."""

    @abstractmethod
    def update_mutables(self, xs, ys):
        """Update the dynamic hyperparameters."""

    def train(self, xs, ys):
        """Train with a dataset."""
        self.update_state(xs, ys)
        self.update_mutables(xs, ys)

    def make_predict(self):
        """Make a predicting function."""
        return self._choose(sigmoid_map, softmax_map)(
            self.model.apply, self.state.params
        )


class StandardTrainer(ContinualTrainer):
    """Standard continual trainer which uses mini-batch gradient descent."""

    def update_state(self, xs, ys):
        """Update the training state."""
        step = make_step(self.mutables['loss'])
        for key in random.split(
            random.PRNGKey(self.immutables['shuffle_seed']),
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
        step = make_step(self.mutables['loss'])
        for key in random.split(
            random.PRNGKey(self.immutables['shuffle_seed']),
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
        step = make_step(self.mutables['loss'])
        n_batches = -(len(ys) // -self.immutables['batch_size'])
        for key1, key2 in zip(
            random.split(
                random.PRNGKey(self.immutables['shuffle_seed']),
                num=self.immutables['n_epochs']
            ),
            random.split(
                random.PRNGKey(self.immutables['choice_seed']),
                num=self.immutables['n_epochs']
            )
        ):
            for (xs_batch, ys_batch), key in zip(
                draw_batches(key1, self.immutables['batch_size'], xs, ys),
                random.split(key2, num=n_batches)
            ):
                indices = (
                    random.choice(
                        key, len(coreset_ys),
                        shape=(self.immutables['batch_size'],),
                        replace=False
                    ) if len(coreset_ys) > 0
                    else []
                )
                self.state = step(
                    self.state,
                    xs_batch, ys_batch,
                    coreset_xs[indices], coreset_ys[indices]
                )
        xs_path.unlink()
        ys_path.unlink()


class Finetuning(StandardTrainer):
    """Fine-tuning for continual learning."""

    def update_mutables(self, xs, ys):
        """Update the mutable hyperparameters."""
