"""Replay."""

from pathlib import Path

import numpy as np
from jax import random
from orbax.checkpoint.test_utils import erase_and_create_empty
import zarr

from dataops.array import pass_batches

from .base import ParallelTrainer, SerialTrainer
from .loss import reg, reg2


class EmptyMixin:
    """Mixin for initializing an empty coreset."""

    def init_coreset(self):
        """Initialize the coreset."""
        path = Path('coreset.zarr').resolve()
        erase_and_create_empty(path)
        group = zarr.open(path, mode='a')
        group['xs'] = np.empty(
            (0, *self.immutables['input_shape']),
            dtype=np.float32
        )
        group['ys'] = np.empty((0,), dtype=np.uint8)
        return group


class NoiseMixin:
    """Mixin for initializing a random coreset."""

    def init_coreset(self):
        """Initialize the coreset."""
        path = Path('coreset.zarr').resolve()
        erase_and_create_empty(path)
        group = zarr.open(path, mode='a')
        group['xs'] = np.asarray(random.uniform(
            random.PRNGKey(self.immutables['init_coreset_seed']),
            shape=(
                self.immutables['coreset_size'],
                *self.immutables['input_shape']
            )
        ))
        group['ys'] = np.asarray(random.choice(
            random.PRNGKey(self.immutables['init_coreset_seed']),
            self.immutables['output_size'],
            shape=(self.immutables['coreset_size'],)
        ))
        return group


class JointMixin:
    """Mixin for joint coreset update."""

    def update_coreset(self, xs, ys):
        """Update the coreset."""
        for xs_batch, ys_batch in pass_batches(
            self.immutables['pass_batch_size'], xs, ys
        ):
            self.mutables['coreset']['xs'].append(xs_batch)
            self.mutables['coreset']['ys'].append(ys_batch)


class GDumbMixin:
    """Mixin for GDumb coreset selection."""

    def update_coreset(self, xs, ys):
        """Update the coreset."""
        for x, y in zip(xs, ys):
            if (
                len(self.mutables['coreset']['ys'])
                < self.immutables['coreset_size']
            ):
                self.mutables['coreset']['xs'].append(np.expand_dims(x, 0))
                self.mutables['coreset']['ys'].append(np.expand_dims(y, 0))
            else:
                key1, key2 = random.split(
                    random.PRNGKey(self.immutables['update_coreset_seed'])
                )
                ys = self.mutables['coreset']['ys'][:]
                count = np.bincount(ys)
                mode = random.choice(
                    key1, (count == count.max()).nonzero()[0]
                )
                index = random.choice(key2, (ys == mode).nonzero()[0]).item()
                self.mutables['coreset']['xs'][index] = x
                self.mutables['coreset']['ys'][index] = y


class Joint(EmptyMixin, JointMixin, SerialTrainer):
    """Joint training."""

    def init_mutables(self):
        """Initialize the mutable hyperparameters."""
        return {
            'loss_fn': reg(
                self.immutables['precision'],
                self.immutables['basic_loss'],
                self.model.apply
            ),
            'coreset': self.init_coreset()
        }

    def update_mutables(self, xs, ys):
        """Update the coreset."""
        self.update_coreset(xs, ys)


class GDumb(EmptyMixin, GDumbMixin, ParallelTrainer):
    """GDumb."""

    def init_mutables(self):
        """Initialize the mutable hyperparameters."""
        return {
            'loss_fn': reg2(
                self.immutables['precision'],
                self.immutables['basic_loss'],
                self.model.apply
            ),
            'coreset': self.init_coreset()
        }

    def update_mutables(self, xs, ys):
        """Update the hyperparameters."""
        self.update_coreset(xs, ys)
