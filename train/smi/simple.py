"""Simple trainers."""

from jax import jit, random

from dataops.array import batch, shuffle

from ..coreset import JointCoreset
from ..loss import basic_loss
from ..state.functions import make_step
from ..state.mixins import MAPMixin
from ..trainer import ContinualTrainer


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
                'coreset.zarr', 'coreset.memmap', self.model_spec
            )
        }

    def update_loss(self, xs, ys):
        """Update the loss function."""
        self.loss = jit(
            basic_loss(
                self.model_spec.fin_act,
                self.immutables['precision'],
                self.model.apply
            )
        )

    def update_state(self, xs, ys):
        """Update the training state."""
        self.mutables['coreset'].update(
            self.precomputed['keys']['update_coreset'], xs, ys
        )
        self.mutables['coreset'].create_memmap()
        step = make_step(self.loss)
        for key in random.split(
            self.precomputed['keys']['update_state'],
            num=self.immutables['n_epochs']
        ):
            key1, key2 = random.split(key)
            for xs_batch, ys_batch in (
                self.mutables['coreset'].shuffle_batch(
                    key2, self.immutables['batch_size']
                )
            ):
                self.state = step(self.state, xs_batch, ys_batch)
            yield self.state
        self.mutables['coreset'].delete_memmap()

    def update_mutables(self, xs, ys):
        """Update mutables."""


class RegularTrainer(MAPMixin, ContinualTrainer):
    """Mixin for regular SGD."""

    def update_state(self, xs, ys):
        """Update the training state."""
        step = make_step(self.loss)
        for key in random.split(
            self.precomputed['keys']['update_state'],
            num=self.immutables['n_epochs']
        ):
            for indices in batch(
                self.immutables['batch_size'], shuffle(key, len(ys))
            ):
                self.state = step(self.state, xs[indices], ys[indices])
            yield self.state


class Finetuning(RegularTrainer):
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
                self.model_spec.fin_act,
                self.immutables['precision'],
                self.model.apply
            )
        )

    def update_mutables(self, xs, ys):
        """Update the mutable hyperparameters."""
