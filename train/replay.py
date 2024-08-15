"""Replay."""

from abc import abstractmethod

import numpy as np
from jax import random
from jax.nn import sigmoid, softmax
from torch.utils.data import ConcatDataset, Subset

from dataio.datasets import ArrayDataset, to_arrays

from . import ContinualTrainer, InitStateMixin, UpdateStateMixin
from . import loss


class InitCoresetMixin:
    """Mixin for initializing a coreset randomly."""

    def _init_coreset(self):
        """Initialize the coreset."""
        xs = np.asarray(random.uniform(
            random.PRNGKey(self.hyperparams['init_coreset_key']),
            shape=(
                self.hyperparams['coreset_size'],
                *self.hyperparams['input_shape'][1:]
            ),
            minval=self.hyperparams['init_coreset_minval'],
            maxval=self.hyperparams['init_coreset_maxval']
        ))
        ys = np.asarray(
            self.make_predictor(params=self.state.params).predict(xs)
        )
        return ArrayDataset(xs, ys)


class FullCoresetMixin:
    """Mixin for making a full coreset storing all data."""

    def _update_coreset(self, dataset):
        """Update the coreset."""
        self.coreset = ConcatDataset([self.coreset, dataset])


class RandomCoresetMixin:
    """Mixin for making a random coreset of fixed size."""

    def _update_coreset(self, dataset):
        """Update the coreset."""
        self.coreset = ConcatDataset([self.coreset, dataset])
        if len(self.coreset) > self.hyperparams['coreset_size']:
            self.coreset = Subset(
                self.coreset,
                np.asarray(random.choice(
                    random.PRNGKey(self.hyperparams['coreset_selection_key']),
                    len(self.coreset),
                    shape=(self.hyperparams['coreset_size'],),
                    replace=False
                ))
            )


class BalancedRandomCoresetMixin:
    """Mixin for making a balanced random coreset of fixed size."""

    def _update_coreset(self, dataset):
        """Update coreset."""
        self.coreset = ConcatDataset([self.coreset, dataset])
        if len(self.coreset) > self.hyperparams['coreset_size']:
            _, ys = to_arrays(
                self.coreset,
                memmap=self.hyperparams['memmap']
            )
            classes, counts = np.unique(ys, return_counts=True)
            y_indices = np.select(
                [ys == c for c in classes],
                np.arange(len(classes)),
                default=ys
            )
            self.coreset = Subset(
                self.coreset,
                np.asarray(random.choice(
                    random.PRNGKey(self.hyperparams['coreset_selection_key']),
                    len(self.coreset),
                    p=(1 / counts)[y_indices],
                    shape=(self.hyperparams['coreset_size'],),
                    replace=False
                ))
            )


class Replay(InitStateMixin, UpdateStateMixin, ContinualTrainer):
    """Abstract class for exact replay."""

    def __init__(self, model, make_predictor, hyperparams):
        """Initialize self."""
        super().__init__(model, make_predictor, hyperparams)
        self.state = self._init_state()
        self.loss_fn = loss.reg(self.hyperparams['precision'], self.basic_loss_fn)
        self.coreset = []

    def train(self, dataset):
        """Train with a dataset."""
        xs, ys = to_arrays(
            ConcatDataset([self.coreset, dataset]),
            memmap=self.hyperparams['memmap']
        )
        self._update_state(xs, ys)
        self._update_coreset(dataset)

    @abstractmethod
    def _update_coreset(self, dataset):
        """Update coreset."""


class Joint(FullCoresetMixin, Replay):
    """Joint training."""


class RandomCoresetReplay(RandomCoresetMixin, Replay):
    """Exact replay with a random coreset."""


class BalancedRandomCoresetReplay(BalancedRandomCoresetMixin, Replay):
    """Exact replay with a random coreset."""
