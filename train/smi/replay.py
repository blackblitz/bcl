"""Sequential MAP inference."""

from jax import jit, random

from dataops.array import batch, shuffle

from ..coreset import GDumbCoreset, TaskIncrementalCoreset
from ..loss import basic_loss, concat_loss
from ..state.functions import make_step
from ..state.mixins import MAPMixin
from ..trainer import ContinualTrainer


class ExactReplay(MAPMixin, ContinualTrainer):
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
                self.model_spec.fin_act,
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
                self.model_spec, self.immutables['coreset_size']
            )
        }

    def update_state(self, xs, ys):
        """Update the training state."""
        self.mutables['coreset'].create_memmap()
        step = make_step(self.loss)
        for key in random.split(
            self.precomputed['keys']['update_state'],
            num=self.immutables['n_epochs']
        ):
            keys = random.split(
                key, num=-(len(ys) // -self.immutables['batch_size']) + 1
            )
            for i, indices in enumerate(
                batch(
                    self.immutables['batch_size'],
                    shuffle(keys[0], len(ys))
                ),
                start=1
            ):
                self.state = step(
                    self.state, xs[indices], ys[indices],
                    *self.mutables['coreset'].choice(
                        keys[i],
                        self.immutables['coreset_batch_size']
                    )
                )
            yield self.state
        self.mutables['coreset'].delete_memmap()


class TICReplay(ExactReplay):
    """GDumb."""

    def init_mutables(self):
        """Initialize the mutable hyperparameters."""
        return {
            'coreset': TaskIncrementalCoreset(
                'coreset.zarr', 'coreset.memmap',
                self.model_spec, self.immutables['coreset_size_per_task']
            )
        }

    def update_state(self, xs, ys):
        """Update the training state."""
        self.mutables['coreset'].create_memmap()
        step = make_step(self.loss)
        for key in random.split(
            self.precomputed['keys']['update_state'],
            num=self.immutables['n_epochs']
        ):
            keys = random.split(
                key, num=-(len(ys) // -self.immutables['batch_size']) + 1
            )
            for i, indices in enumerate(
                batch(
                    self.immutables['batch_size'],
                    shuffle(keys[0], len(ys))
                ),
                start=1
            ):
                self.state = step(
                    self.state, xs[indices], ys[indices],
                    *self.mutables['coreset'].choice(
                        keys[i],
                        self.immutables['coreset_batch_size_per_task']
                    )
                )
            yield self.state
        self.mutables['coreset'].delete_memmap()
