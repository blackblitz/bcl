"""Base classes."""

from abc import ABC, abstractmethod
from itertools import chain
from pathlib import Path

from flax.training.train_state import TrainState
from jax import grad, jit, random
import numpy as np
from optax import adam

from dataops.array import draw_batches
from dataops.io import zarr_to_memmap

from .loss import reg


def make_step(loss_fn):
    """Make a gradient-descent step function for a loss function."""
    return jit(
        lambda state, *args: state.apply_gradients(
            grads=grad(loss_fn)(state.params, *args)
        )
    )


class ContinualTrainer(ABC):
    """Abstract base class for continual learning."""

    def __init__(self, model, immutables):
        """Intialize self."""
        self.model = model
        self.immutables = immutables
        self.state = self.init_state()
        self.mutables = self.init_mutables()

    def init_state(self):
        """Initialize the state."""
        return TrainState.create(
            apply_fn=self.model.apply,
            params=self.model.init(
                random.PRNGKey(self.immutables['init_state_seed']),
                np.zeros((1, *self.immutables['input_shape']))
            )['params'],
            tx=adam(self.immutables['lr'])
        )

    def init_mutables(self):
        """Initialize the mutable hyperparameters."""
        return {
            'loss_fn': reg(
                self.immutables['precision'],
                self.immutables['basic_loss'],
                self.model.apply
            )
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


class StandardTrainer(ContinualTrainer):
    """Standard continual trainer which uses mini-batch gradient descent."""

    def update_state(self, xs, ys):
        """Update the training state."""
        step = make_step(self.mutables['loss_fn'])
        for key in random.split(
            random.PRNGKey(self.immutables['shuffle_seed']),
            num=self.immutables['n_epochs']
        ):
            for xs_batch, ys_batch in draw_batches(
                key, self.immutables['draw_batch_size'], xs, ys
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
        step = make_step(self.mutables['loss_fn'])
        for key in random.split(
            random.PRNGKey(self.immutables['shuffle_seed']),
            num=self.immutables['n_epochs']
        ):
            for xs_batch, ys_batch in chain(
                draw_batches(
                    key, self.immutables['draw_batch_size'], xs, ys
                ),
                draw_batches(
                    key, self.immutables['draw_batch_size'],
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
        step = make_step(self.mutables['loss_fn'])
        n_batches = -(len(ys) // -self.immutables['draw_batch_size'])
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
                draw_batches(
                    key1, self.immutables['draw_batch_size'], xs, ys
                ),
                random.split(key2, num=n_batches)
            ):
                indices = (
                    random.choice(
                        key, len(coreset_ys),
                        shape=(self.immutables['draw_batch_size'],),
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
