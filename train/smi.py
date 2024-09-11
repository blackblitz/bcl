"""Sequential MAP inference."""

from jax import jit

from .base import ContinualTrainer
from .coreset import GDumbCoreset, JointCoreset, TaskIncrementalCoreset
from .loss import basic_loss, concat_loss
from .state.mixins import (
    MAPMixin, ParallelShuffleMixin, RegularMixin, SerialMixin
)


class Finetuning(MAPMixin, RegularMixin, ContinualTrainer):
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
            basic_loss(
                self.immutables['nntype'],
                self.immutables['precision'],
                self.model.apply
            )
        )

    def update_mutables(self, xs, ys):
        """Update the mutable hyperparameters."""


class Joint(MAPMixin, SerialMixin, ContinualTrainer):
    """Joint training."""

    def precompute(self):
        """Precompute."""
        return super().precompute() | self._make_keys(
            ['init_state', 'update_state', 'update_coreset']
        )

    def init_mutables(self):
        """Initialize the mutable hyperparameters."""
        return {
            'coreset': JointCoreset(
                'coreset.zarr', 'coreset.memmap',
                self.immutables, self.metadata
            )
        }

    def update_loss(self, xs, ys):
        """Update the loss function."""
        self.loss = jit(
            basic_loss(
                self.immutables['nntype'],
                self.immutables['precision'],
                self.model.apply
            )
        )

    def update_mutables(self, xs, ys):
        """Update the coreset."""
        self.mutables['coreset'].update(
            self.precomputed['keys']['update_coreset'], xs, ys
        )


class ExactReplay(MAPMixin, ParallelShuffleMixin, ContinualTrainer):
    """Exact replay."""
    def precompute(self):
        """Precompute."""
        return super().precompute() | self._make_keys(
            ['init_state', 'update_state', 'update_coreset']
        )

    def update_loss(self, xs, ys):
        """Update the loss function."""
        self.loss = jit(concat_loss(
            basic_loss(
                self.immutables['nntype'],
                self.immutables['precision'],
                self.model.apply
            )
        ))

    def update_mutables(self, xs, ys):
        """Update the coreset."""
        self.mutables['coreset'].update(
            self.precomputed['keys']['update_coreset'], xs, ys
        )


class GDumb(ExactReplay):
    """GDumb."""

    def init_mutables(self):
        """Initialize the mutable hyperparameters."""
        return {
            'coreset': GDumbCoreset(
                'coreset.zarr', 'coreset.memmap',
                self.immutables, self.metadata
            )
        }


class TICReplay(ExactReplay):
    """GDumb."""

    def init_mutables(self):
        """Initialize the mutable hyperparameters."""
        return {
            'coreset': TaskIncrementalCoreset(
                'coreset.zarr', 'coreset.memmap',
                self.immutables, self.metadata
            )
        }
