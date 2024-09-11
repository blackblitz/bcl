"""Replay."""

from jax import jit

from .base import ContinualTrainer, MAPMixin
from .coreset import GDumbCoreset, JointCoreset, TaskIncrementalCoreset
from .loss import concat_loss, sigmoid_ce, softmax_ce
from .state import (
    regular_sgd, serial_sgd, parallel_sgd_choice, parallel_sgd_shuffle_batch
)


class Joint(MAPMixin, ContinualTrainer):
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
            self._choose(sigmoid_ce, softmax_ce)(
                self.immutables['precision'], self.model.apply
            )
        )

    def update_state(self, xs, ys):
        """Update the training state."""
        self.state = serial_sgd(
            self.precomputed['keys']['update_state'],
            self.immutables['n_epochs'],
            self.immutables['batch_size'],
            self.loss,
            self.state,
            xs, ys, self.mutables['coreset']
        )

    def update_mutables(self, xs, ys):
        """Update the coreset."""
        self.mutables['coreset'].update(
            self.precomputed['keys']['update_coreset'], xs, ys
        )


class ExactReplay(MAPMixin, ContinualTrainer):
    """Exact replay."""
    def precompute(self):
        """Precompute."""
        return super().precompute() | self._make_keys(
            ['init_state', 'update_state', 'update_coreset']
        )

    def update_loss(self, xs, ys):
        """Update the loss function."""
        self.loss = jit(concat_loss(
            self._choose(sigmoid_ce, softmax_ce)(
                self.immutables['precision'], self.model.apply
            )
        ))

    def update_state(self, xs, ys):
        """Update the training state."""
        self.state = parallel_sgd_shuffle_batch(
            self.precomputed['keys']['update_state'],
            self.immutables['n_epochs'],
            self.immutables['batch_size'],
            self.loss,
            self.state,
            xs, ys, self.mutables['coreset']
        )

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
