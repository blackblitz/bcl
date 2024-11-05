"""Sequential MAP inference."""

from jax import jit, random

from dataops.array import batch, shuffle

from . import MAPMixin
from .. import ContinualTrainer
from ..coreset import GDumbCoreset
from ...training import make_step
from ...training.loss import concat, l2_reg


class ExactReplay(MAPMixin, ContinualTrainer):
    """Abstract class for exact replay."""

    def update_loss(self, xs, ys):
        """Update the loss function."""
        self.loss = jit(concat(l2_reg(
            self.hparams['precision'], self.hparams['nll']
        )))

    def update_hparams(self, xs, ys):
        """Update the coreset."""
        self.hparams['coreset'].update(
            self.hparams['keys']['update_hparams'], xs, ys
        )


class GDumb(ExactReplay):
    """GDumb."""

    def __init__(self, model, mspec, hparams):
        """Initialize self."""
        super().__init__(model, mspec, hparams)
        self.hparams['coreset'] = GDumbCoreset(
            'coreset.zarr', 'coreset.memmap',
            self.mspec, self.hparams['coreset_size']
        )

    def update_state(self, xs, ys):
        """Update the training state."""
        key1, key2 = random.split(self.hparams['keys']['update_state'])
        self.hparams['coreset'].create_memmap()
        step = make_step(self.loss)
        self.state = self._init_state(key1, len(ys))
        for key in random.split(key2, num=self.hparams['n_epochs']):
            keys = random.split(
                key, num=-(len(ys) // -self.hparams['batch_size']) + 1
            )
            for i, indices in enumerate(
                batch(
                    self.hparams['batch_size'],
                    shuffle(keys[0], len(ys))
                ),
                start=1
            ):
                self.state = step(
                    self.state, xs[indices], ys[indices],
                    *self.hparams['coreset'].choice(
                        keys[i],
                        self.hparams['coreset_batch_size']
                    )
                )
            yield self.state
        self.hparams['coreset'].delete_memmap()
