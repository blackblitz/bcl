"""Replay."""

from pathlib import Path

import numpy as np
from jax import random
from torch.utils.data import ConcatDataset, Subset

from dataio.dataset_sequences.datasets import ArrayDataset, dataset_to_arrays

from .base import ConcatTrainer
from .loss import reg, reg_duo


class InitCoresetMixin:
    """Mixin for initializing a coreset randomly."""

    def init_coreset(self):
        """Initialize the coreset."""
        xs = np.asarray(random.uniform(
            random.PRNGKey(self.immutables['init_coreset_seed']),
            shape=(
                self.immutables['coreset_size'],
                *self.immutables['input_shape'][1:]
            ),
            minval=self.immutables['init_coreset_minval'],
            maxval=self.immutables['init_coreset_maxval']
        ))
        ys = np.asarray(
            self.make_predictor(params=self.state.params).predict(xs)
        )
        return ArrayDataset(xs, ys)


class RandomCoresetMixin:
    """Mixin for making a random coreset of fixed size."""

    def update_coreset(self, dataset):
        """Update coreset."""
        self.mutables['coreset'] = ConcatDataset(
            [self.mutables['coreset'], dataset]
        )
        if (
            len(self.mutables['coreset'])
            > self.immutables['coreset_size']
        ):
            path = Path('data.npy')
            _, ys = dataset_to_arrays(
                self.mutables['coreset'], path
            )
            path.unlink()
            classes, counts = np.unique(ys, return_counts=True)
            y_indices = np.select(
                [ys == c for c in classes],
                np.arange(len(classes)),
                default=ys
            )
            self.mutables['coreset'] = Subset(
                self.mutables['coreset'],
                np.asarray(random.choice(
                    random.PRNGKey(
                        self.immutables['coreset_selection_seed']
                    ),
                    len(self.mutables['coreset']),
                    p=(1 / counts)[y_indices],
                    shape=(self.immutables['coreset_size'],),
                    replace=False
                ))
            )


class Joint(ConcatTrainer):
    """Joint training."""

    def init_mutables(self):
        """Initialize the mutable hyperparameters."""
        return {
            'loss_fn': reg(
                self.immutables['precision'],
                self.immutables['basic_loss'],
                self.model.apply
            ),
            'coreset': []
        }

    def update_mutables(self, dataset):
        """Update the coreset."""
        self.mutables['coreset'] = ConcatDataset(
            [self.mutables['coreset'], dataset]
        )


class RandomConcatReplay(RandomCoresetMixin, ConcatTrainer):
    """Exact replay with a random coreset by concatenating."""

    def init_mutables(self):
        """Initialize the mutable hyperparameters."""
        return {
            'loss_fn': reg(
                self.immutables['precision'],
                self.immutables['basic_loss'],
                self.model.apply
            ),
            'coreset': []
        }

    def update_mutables(self, dataset):
        """Update the hyperparameters."""
        self.update_coreset(dataset)


class RandomChoiceReplay(RandomCoresetMixin, ChoiceTrainer):
    """Exact replay with a random coreset by choice."""

    def init_mutables(self):
        """Initialize the mutable hyperparameters."""
        return {
            'loss_fn': reg_duo(
                self.immutables['precision'],
                self.immutables['basic_loss'],
                self.model.apply
            ),
            'coreset': []
        }

    def update_mutables(self, dataset):
        """Update the hyperparameters."""
        self.update_coreset(dataset)
