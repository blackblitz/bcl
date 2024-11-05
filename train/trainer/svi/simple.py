"""Simple trainers."""

from jax import jit, random
import jax.numpy as jnp

from dataops.array import batch, get_n_batches, shuffle

from . import GaussMixin, GaussmixMixin, SVI, TMixin
from ...training.loss import (
    gpvfe_cf, gpvfe_mc, gfvfe_cf, gfvfe_mc, tpvfe_mc, tfvfe_mc,
    gmpvfe_ub, gmpvfe_mc, gmfvfe_ub, gmfvfe_mc,
)
from ...training import make_step


class VCL(SVI):
    """Variational continual learning."""

    def update_state(self, xs, ys):
        """Update the training state."""
        key1, key2 = random.split(self.hparams['keys']['update_state'])
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
                self.state = step(
                    self.state, self._sample(subkey),
                    xs[indices], ys[indices]
                )
            yield self.state


class GVCL(GaussMixin, VCL):
    """Gaussian variational continual learning."""

    def update_loss(self, xs, ys):
        """Update the loss function."""
        n_batches = -(len(ys) // -self.hparams['batch_size'])
        self.loss = jit(
            (gpvfe_mc if self.hparams['mc'] else gpvfe_cf)(
                self.hparams['nll'],
                self.hparams.get('beta', 1 / n_batches),
                self.hparams['prior']
            )
        )


class TVCL(TMixin, VCL):
    """t variational continual learning."""

    def update_loss(self, xs, ys):
        """Update the loss function."""
        n_batches = -(len(ys) // -self.hparams['batch_size'])
        self.loss = jit(
            tpvfe_mc(
                self.hparams['nll'],
                self.hparams.get('beta', 1 / n_batches),
                self.hparams['prior'],
                self.hparams['df']
            )
        )


class GMVCL(GaussmixMixin, VCL):
    """Gaussian-mixture variational continual learning."""

    def update_loss(self, xs, ys):
        """Update the loss function."""
        n_batches = -(len(ys) // -self.hparams['batch_size'])
        self.loss = jit(
            (gmpvfe_mc if self.hparams['mc'] else gmpvfe_ub)(
                self.hparams['nll'],
                self.hparams.get('beta', 1 / n_batches),
                self.hparams['prior']
            )
        )


class SimpleSFSVI(SVI):
    """Simple S-FSVI."""

    def update_state(self, xs, ys):
        """Update the training state."""
        key1, key2 = random.split(self.hparams['keys']['update_state'])
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
                subkeys = random.split(subkey, num=3)
                param_sample = self._sample(subkeys[0])
                ind_xs = random.uniform(
                    subkeys[1],
                    shape=(
                        self.hparams['noise_batch_size'],
                        *self.mspec.in_shape
                    ),
                    minval=jnp.array(self.hparams['noise_minval']),
                    maxval=jnp.array(self.hparams['noise_maxval']),
                )
                output_sample = self._sample(subkeys[2], xs=ind_xs)
                self.state = step(
                    self.state, param_sample, output_sample,
                    xs[indices], ys[indices], ind_xs
                )
            yield self.state


class SimpleGSFSVI(GaussMixin, SimpleSFSVI):
    """Simple Gaussian S-FSVI."""

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


class SimpleTSFSVI(TMixin, SimpleSFSVI):
    """Simple t S-FSVI."""

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


class SimpleGMSFSVI(GaussmixMixin, SimpleSFSVI):
    """Simple Gaussian-mixture S-FSVI."""

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
