"""Sequential MAP inference."""

from jax import jit, random

from dataops.array import batch, shuffle

from ..base import ContinualTrainer, MAPMixin
from ..coreset import GDumbCoreset, TaskIncrementalCoreset
from ...training.loss.stateless import concat, l2_reg
from ...training.stateless import make_step


class ExactReplay(MAPMixin, ContinualTrainer):
    """Abstract class for exact replay."""

    def update_loss(self, xs, ys):
        """Update the loss function."""
        self.loss = jit(concat(l2_reg(
            self.immutables['precision'], self.precomputed['nll']
        )))

    def update_mutables(self, xs, ys):
        """Update the coreset."""
        self.mutables['coreset'].update(
            self.precomputed['keys']['update_mutables'], xs, ys
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
