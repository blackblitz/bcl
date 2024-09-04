"""Replay."""

from pathlib import Path

import numpy as np
from jax import random
from orbax.checkpoint.test_utils import erase_and_create_empty
import zarr

from dataops.array import pass_batches

from .base import ParallelTrainer, SerialTrainer
from .loss import concat_loss, sigmoid_ce, softmax_ce


class EmptyMixin:
    """Mixin for initializing an empty coreset."""

    def init_coreset(self):
        """Initialize the coreset."""
        path = Path('coreset.zarr').resolve()
        erase_and_create_empty(path)
        group = zarr.open(path, mode='a')
        group['xs'] = np.empty(
            (0, *self.metadata['input_shape']),
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
        key1, key2 = random.split(self.precomputed['keys']['init_coreset'])
        group['xs'] = np.asarray(random.uniform(
            key1, shape=(
                self.immutables['coreset_size'],
                *self.metadata['input_shape']
            )
        ))
        group['ys'] = np.asarray(random.choice(
            key2, len(self.metadata['classes']),
            shape=(self.immutables['coreset_size'],)
        ))
        return group


class JointMixin:
    """Mixin for joint coreset update."""

    def update_coreset(self, xs, ys):
        """Update the coreset."""
        for xs_batch, ys_batch in pass_batches(
            self.precomputed['pass_size'], xs, ys
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
                    self.precomputed['keys']['update_coreset']
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
    def precompute(self):
        """Precompute."""
        return super().precompute() | self._make_keys(
            ['init_state', 'update_state']
        )

    def init_mutables(self):
        """Initialize the mutable hyperparameters."""
        return {
            'loss': self._choose(sigmoid_ce, softmax_ce)(
                self.immutables['precision'], self.model.apply
            ),
            'coreset': self.init_coreset()
        }

    def update_mutables(self, xs, ys):
        """Update the coreset."""
        self.update_coreset(xs, ys)


class GDumb(EmptyMixin, GDumbMixin, ParallelTrainer):
    """GDumb."""
    def precompute(self):
        """Precompute."""
        return super().precompute() | self._make_keys(
            ['init_state', 'update_state', 'update_coreset']
        )

    def init_mutables(self):
        """Initialize the mutable hyperparameters."""
        return {
            'loss': concat_loss(self._choose(sigmoid_ce, softmax_ce)(
                self.immutables['precision'], self.model.apply
            )),
            'coreset': self.init_coreset()
        }

    def update_mutables(self, xs, ys):
        """Update the hyperparameters."""
        self.update_coreset(xs, ys)
