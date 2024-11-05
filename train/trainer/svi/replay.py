"""Replay."""

from jax import jit, random
import jax.numpy as jnp

from dataops.array import batch, get_n_batches, shuffle

from . import GaussMixin, GaussmixMixin, SVI, TMixin
from ..coreset import TaskIncrementalCoreset
from ...training.loss import (
    gfvfe_cf, gfvfe_mc, tfvfe_mc, gmfvfe_ub, gmfvfe_mc
)
from ...training import make_step
from ...training.vi import gauss, gaussmix, t


class PriorExactSFSVI(SVI):
    """Prior-focused exact-replay S-FSVI."""

    def __init__(self, model, mspec, hparams):
        """Initialize the mutable hyperparameters."""
        super().__init__(model, mspec, hparams)
        self.hparams['coreset'] = TaskIncrementalCoreset(
            'coreset.zarr', 'coreset.memmap',
            self.mspec, self.hparams['coreset_size_per_task']
        )

    def update_state(self, xs, ys):
        """Update the training state."""
        key1, key2 = random.split(self.hparams['keys']['update_state'])
        self.hparams['coreset'].create_memmap()
        step = make_step(self.loss)
        self.state = self._init_state(key1, len(ys))
        for key in random.split(key2, num=self.hparams['n_epochs']):
            n_batches = get_n_batches(len(ys), self.hparams['batch_size'])
            keys = random.split(key, num=n_batches + 1)
            for subkey, indices in zip(
                keys[: n_batches],
                batch(
                    self.hparams['batch_size'],
                    shuffle(keys[n_batches], len(ys))
                )
            ):
                subkeys = random.split(subkey, num=4)
                param_sample = self._sample(subkeys[0])
                if (
                    self.hparams['coreset'].task_count > 0
                    and random.bernoulli(
                        subkeys[1],
                        p=self.hparams['coreset_prob']
                    )
                ):
                    ind_xs, _ = self.hparams['coreset'].choice(
                        subkeys[2],
                        self.hparams['coreset_batch_size_per_task']
                    )
                else:
                    ind_xs = random.uniform(
                        subkeys[2],
                        shape=(
                            self.hparams['coreset_batch_size_per_task'],
                            *self.mspec.in_shape
                        ),
                        minval=jnp.array(self.hparams['noise_minval']),
                        maxval=jnp.array(self.hparams['noise_maxval']),
                    )
                output_sample = self._sample(subkeys[3], xs=ind_xs)
                self.state = step(
                    self.state, param_sample, output_sample,
                    xs[indices], ys[indices], ind_xs
                )
            yield self.state
        self.hparams['coreset'].delete_memmap()

    def update_hparams(self, xs, ys):
        """Update the hyperparameters."""
        self.hparams['prior'] = self.state.params
        self.hparams['coreset'].update(
            self.hparams['keys']['update_hparams'], xs, ys
        )


class PriorExactGSFSVI(GaussMixin, PriorExactSFSVI):
    """Prior-focused exact-replay Gaussian S-FSVI."""

    def update_loss(self, xs, ys):
        """Update the loss function."""
        n_batches = -(len(ys) // -self.hparams['batch_size'])
        self.loss = jit(
            (gfvfe_mc if self.hparams['mc'] else gfvfe_cf)(
                self.hparams['nll'],
                self.hparams.get('beta', 1 / n_batches),
                self.hparams['prior'],
                self.model.apply
            )
        )


class PriorExactTSFSVI(TMixin, PriorExactSFSVI):
    """Prior-focused exact-replay Student's t S-FSVI."""

    def update_loss(self, xs, ys):
        """Update the loss function."""
        n_batches = -(len(ys) // -self.hparams['batch_size'])
        self.loss = jit(
            tfvfe_mc(
                self.hparams['nll'],
                self.hparams.get('beta', 1 / n_batches),
                self.hparams['prior'],
                self.model.apply,
                self.hparams['df']
            )
        )


class PriorExactGMSFSVI(GaussmixMixin, PriorExactSFSVI):
    """Prior-focused exact-replay Gaussian-mixture S-FSVI."""

    def update_loss(self, xs, ys):
        """Update the loss function."""
        n_batches = -(len(ys) // -self.hparams['batch_size'])
        self.loss = jit(
            (gmfvfe_mc if self.hparams['mc'] else gmfvfe_ub)(
                self.hparams['nll'],
                self.hparams.get('beta', 1 / n_batches),
                self.hparams['prior'],
                self.model.apply
            )
        )
