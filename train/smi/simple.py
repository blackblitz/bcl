"""Simple trainers."""

from jax import jit

from ..loss import basic_loss
from ..state.mixins import MAPMixin, RegularMixin
from ..trainer import ContinualTrainer


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
