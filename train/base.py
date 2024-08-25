"""Base classes."""

from abc import ABC, abstractmethod
from pathlib import Path

from flax.training.train_state import TrainState
from jax import grad, jit, random
import numpy as np
from optax import adam
from torch.utils.data import ConcatDataset

from dataio import draw_batches
from dataio.dataset_sequences.datasets import dataset_to_arrays

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

    def __init__(self, model, make_predictor, immutables):
        """Intialize self."""
        self.model = model
        self.make_predictor = make_predictor
        self.immutables = immutables
        self.state = self.init_state()
        self.mutables = self.init_mutables()

    def init_state(self):
        """Initialize the state."""
        return TrainState.create(
            apply_fn=self.model.apply,
            params=self.model.init(
                random.PRNGKey(self.immutables['init_state_seed']),
                np.zeros(self.immutables['input_shape'])
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
    def update_state(self, dataset):
        """Update the training state."""

    @abstractmethod
    def update_mutables(self, dataset):
        """Update the dynamic hyperparameters."""

    def train(self, dataset):
        """Train with a dataset."""
        self.update_state(dataset)
        self.update_mutables(dataset)


class MinibatchTrainer(ContinualTrainer):
    """Continual trainer that uses mini-batch gradient descent."""

    def update_state(self, dataset):
        """Update the training state."""
        path = Path('data.npy')
        xs, ys = dataset_to_arrays(dataset, path)
        step = make_step(self.mutables['loss_fn'])
        for key in random.split(
            random.PRNGKey(self.immutables['shuffle_seed']),
            num=self.immutables['n_epochs']
        ):
            for xs_batch, ys_batch in draw_batches(
                key, self.immutables['draw_batch_size'], xs, ys
            ):
                self.state = step(self.state, xs_batch, ys_batch)
        path.unlink()


class ConcatTrainer(ContinualTrainer):
    """Continual trainer that uses a coreset by concatenation."""

    def update_state(self, dataset):
        """Update the training state."""
        path = Path('data.npy')
        xs, ys = dataset_to_arrays(
            ConcatDataset([self.mutables['coreset'], dataset]), path
        )
        step = make_step(self.mutables['loss_fn'])
        for key in random.split(
            random.PRNGKey(self.immutables['shuffle_seed']),
            num=self.immutables['n_epochs']
        ):
            for xs_batch, ys_batch in draw_batches(
                key, self.immutables['draw_batch_size'], xs, ys
            ):
                self.state = step(self.state, xs_batch, ys_batch)
        path.unlink()


class ChoiceTrainer(ContinualTrainer):
    """Continual trainer that uses a coreset by choice."""

    def update_state(self, dataset):
        """Update the training state."""
        path = Path('data.npy')
        xs, ys = dataset_to_arrays(dataset, path)
        xs_coreset, ys_coreset = dataset_to_arrays(self.coreset, path)
        step = make_step(self.mutables['loss_fn'])
        n_batches = -(len(dataset) // -self.immutables['draw_batch_size'])
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
                indices = random.choice(key, len(ys_coreset), replace=False)
                self.state = step(
                    self.state,
                    xs_batch, ys_batch,
                    xs_coreset[indices], ys_coreset[indices]
                )
        path.unlink()


class Finetuning(MinibatchTrainer):
    """Fine-tuning for continual learning."""

    def update_mutables(self, dataset):
        """Update the mutable hyperparameters."""
