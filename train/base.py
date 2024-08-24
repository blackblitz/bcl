"""Base classes."""

from abc import ABC, abstractmethod
from pathlib import Path

from flax.training.train_state import TrainState
from jax import grad, jit, random
import numpy as np
from optax import adam

from dataio import draw_batches
from dataio.dataset_sequences.datasets import dataset_to_arrays

from . import loss


def make_step(loss_fn):
    """Make a gradient-descent step function for a loss function."""
    return jit(
        lambda state, *args: state.apply_gradients(
            grads=grad(loss_fn)(state.params, *args)
        )
    )


class ContinualTrainer(ABC):
    """Abstract base class for continual learning."""

    def __init__(self, model, make_predictor, hyperparams):
        """Intialize self."""
        self.model = model
        self.make_predictor = make_predictor
        self.hyperparams = hyperparams
        self.basic_loss_fn = getattr(
            loss,
            self.hyperparams['basic_loss_fn']
        )(self.model.apply)

    @abstractmethod
    def train(self, dataset):
        """Train with a dataset."""


class InitStateMixin:
    """Mixin for initializing the state for MAP prediction."""

    def _init_state(self):
        """Initialize the state."""
        return TrainState.create(
            apply_fn=self.model.apply,
            params=self.model.init(
                random.PRNGKey(self.hyperparams['init_state_seed']),
                np.zeros(self.hyperparams['input_shape'])
            )['params'],
            tx=adam(self.hyperparams['lr'])
        )


class UpdateStateMixin:
    """Mixin for updating the state."""

    def _update_state(self, xs, ys):
        """Update the state."""
        step = make_step(self.loss_fn)
        for key in random.split(
            random.PRNGKey(self.hyperparams['shuffle_seed']),
            num=self.hyperparams['n_epochs']
        ):
            for xs_batch, ys_batch in draw_batches(
                key, self.hyperparams['draw_batch_size'], xs, ys
            ):
                self.state = step(self.state, xs_batch, ys_batch)


class Finetuning(ContinualTrainer, InitStateMixin, UpdateStateMixin):
    """Fine-tuning for continual learning."""

    def __init__(self, model, make_predictor, hyperparams):
        """Initialize self."""
        super().__init__(model, make_predictor, hyperparams)
        self.state = self._init_state()
        self.loss_fn = loss.reg(
            self.hyperparams['precision'], self.basic_loss_fn
        )

    def train(self, dataset):
        """Train with a dataset."""
        path = Path('data.npy')
        xs, ys = dataset_to_arrays(dataset, path)
        self._update_state(xs, ys)
        path.unlink()
