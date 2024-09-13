"""Sequential MAP inference."""

from jax import jit

from ..coreset import GDumbCoreset, JointCoreset, TaskIncrementalCoreset
from ..loss import basic_loss, concat_loss
from ..state.mixins import MAPMixin, ParallelChoiceMixin, SerialMixin
from ..trainer import ContinualTrainer


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


class ExactReplay(MAPMixin, ParallelChoiceMixin, ContinualTrainer):
    """Abstract class for exact replay."""

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
