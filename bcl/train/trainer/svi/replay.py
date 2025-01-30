"""Replay."""

from jax import jit, random
import jax.numpy as jnp


from . import GaussMixin, GaussmixMixin, SVI, TMixin

from .. import coreset_memmap_path, coreset_zarr_path
from ..coreset import TaskIncrementalCoreset
from ...training.loss import (
    concat_fvfe, concat_pvfe, gfvfe_cf, gfvfe_mc,
    gmfvfe_ub, gmfvfe_mc, gmpvfe_ub, gmpvfe_mc,
    gpvfe_cf, gpvfe_mc, tfvfe_mc, tpvfe_mc
)
from ...training import make_step

from ....dataops.array import batch, get_n_batches, shuffle


class PriorExactSFSVI(SVI):
    """Prior-focused exact-replay S-FSVI."""

    def __init__(self, model, mspec, hparams):
        """Initialize the mutable hyperparameters."""
        super().__init__(model, mspec, hparams)
        self.hparams['coreset'] = TaskIncrementalCoreset(
            coreset_memmap_path, coreset_zarr_path,
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
                        self.hparams['coreset_batch_size']
                    )
                else:
                    ind_xs = random.uniform(
                        subkeys[2],
                        shape=(
                            self.hparams['coreset_batch_size'],
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
        n_batches = get_n_batches(len(ys), self.hparams['batch_size'])
        self.loss = jit(
            (gfvfe_mc if self.hparams['mc'] else gfvfe_cf)(
                self.hparams['nll'],
                self.hparams['batch_size']
                / (
                    (self.hparams['coreset'].task_count + 1)
                    * self.hparams['coreset_batch_size']
                ) if self.hparams['equal_weight']
                else self.hparams.get('beta', 1.0) / n_batches,
                self.hparams['prior'],
                self.model.apply
            )
        )


class PriorExactTSFSVI(TMixin, PriorExactSFSVI):
    """Prior-focused exact-replay Student's t S-FSVI."""

    def update_loss(self, xs, ys):
        """Update the loss function."""
        n_batches = get_n_batches(len(ys), self.hparams['batch_size'])
        self.loss = jit(
            tfvfe_mc(
                self.hparams['nll'],
                self.hparams['batch_size']
                / (
                    (self.hparams['coreset'].task_count + 1)
                    * self.hparams['coreset_batch_size']
                ) if self.hparams['equal_weight']
                else self.hparams.get('beta', 1.0) / n_batches,
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
                self.hparams['batch_size']
                / (
                    (self.hparams['coreset'].task_count + 1)
                    * self.hparams['coreset_batch_size']
                ) if self.hparams['equal_weight']
                else self.hparams.get('beta', 1.0) / n_batches,
                self.hparams['prior'],
                self.model.apply
            )
        )


class LikelihoodExactVCL(SVI):
    """Likelihood-focused exact-replay VCL."""

    def __init__(self, model, mspec, hparams):
        """Initialize the mutable hyperparameters."""
        super().__init__(model, mspec, hparams)
        self.hparams['coreset'] = TaskIncrementalCoreset(
            coreset_memmap_path, coreset_zarr_path,
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
                subkeys = random.split(subkey)
                xs2, ys2 = self.hparams['coreset'].choice(
                    subkeys[0],
                    self.hparams['coreset_batch_size']
                )
                self.state = step(
                    self.state, self._sample(subkeys[1]),
                    xs[indices], ys[indices], xs2, ys2
                )
            yield self.state
        self.hparams['coreset'].delete_memmap()

    def update_hparams(self, xs, ys):
        """Update the hyperparameters."""
        self.hparams['coreset'].update(
            self.hparams['keys']['update_hparams'], xs, ys
        )


class LikelihoodExactGVCL(GaussMixin, LikelihoodExactVCL):
    """Likelihood-focused exact-replay Gaussian VCL."""

    def update_loss(self, xs, ys):
        """Update the loss function."""
        n_batches = get_n_batches(len(ys), self.hparams['batch_size'])
        self.loss = jit(concat_pvfe(
            (gpvfe_mc if self.hparams['mc'] else gpvfe_cf)(
                self.hparams['nll'],
                self.hparams.get('beta', 1.0) / n_batches,
                self.hparams['prior']
            )
        ))


class LikelihoodExactTVCL(TMixin, LikelihoodExactVCL):
    """Likelihood-focused exact-replay t VCL."""

    def update_loss(self, xs, ys):
        """Update the loss function."""
        n_batches = get_n_batches(len(ys), self.hparams['batch_size'])
        self.loss = jit(concat_pvfe(
            tpvfe_mc(
                self.hparams['nll'],
                self.hparams.get('beta', 1.0) / n_batches,
                self.hparams['prior'],
                self.hparams['df']
            )
        ))


class LikelihoodExactGMVCL(GaussmixMixin, LikelihoodExactVCL):
    """Likelihood-focused exact-replay Gaussian mixture VCL."""

    def update_loss(self, xs, ys):
        """Update the loss function."""
        n_batches = get_n_batches(len(ys), self.hparams['batch_size'])
        self.loss = jit(concat_pvfe(
            (gmpvfe_mc if self.hparams['mc'] else gmpvfe_ub)(
                self.hparams['nll'],
                self.hparams.get('beta', 1.0) / n_batches,
                self.hparams['prior']
            )
        ))


class LikelihoodExactSFSVI(SVI):
    """Likelihood-focused exact-replay S-FSVI."""

    def __init__(self, model, mspec, hparams):
        """Initialize the mutable hyperparameters."""
        super().__init__(model, mspec, hparams)
        self.hparams['coreset'] = TaskIncrementalCoreset(
            coreset_memmap_path, coreset_zarr_path,
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
                xs2, ys2 = self.hparams['coreset'].choice(
                    subkeys[1],
                    self.hparams['coreset_batch_size']
                )
                ind_xs = random.uniform(
                    subkeys[2],
                    shape=(
                        self.hparams['noise_batch_size'],
                        *self.mspec.in_shape
                    ),
                    minval=jnp.array(self.hparams['noise_minval']),
                    maxval=jnp.array(self.hparams['noise_maxval']),
                )
                output_sample = self._sample(subkeys[3], xs=ind_xs)
                self.state = step(
                    self.state, param_sample, output_sample,
                    xs[indices], ys[indices], xs2, ys2, ind_xs
                )
            yield self.state
        self.hparams['coreset'].delete_memmap()

    def update_hparams(self, xs, ys):
        """Update the hyperparameters."""
        self.hparams['coreset'].update(
            self.hparams['keys']['update_hparams'], xs, ys
        )


class LikelihoodExactGSFSVI(GaussMixin, LikelihoodExactSFSVI):
    """Likelihood-focused exact-replay Gaussian S-FSVI."""

    def update_loss(self, xs, ys):
        """Update the loss function."""
        n_batches = get_n_batches(len(ys), self.hparams['batch_size'])
        self.loss = jit(concat_fvfe(
            (gfvfe_mc if self.hparams['mc'] else gfvfe_cf)(
                self.hparams['nll'],
                self.hparams['batch_size'] / self.hparams['noise_batch_size']
                if self.hparams['equal_weight']
                else self.hparams.get('beta', 1.0) / n_batches,
                self.hparams['prior'],
                self.model.apply
            )
        ))


class LikelihoodExactTSFSVI(TMixin, LikelihoodExactSFSVI):
    """Likelihood-focused exact-replay Student's t S-FSVI."""

    def update_loss(self, xs, ys):
        """Update the loss function."""
        n_batches = get_n_batches(len(ys), self.hparams['batch_size'])
        self.loss = jit(concat_fvfe(
            tfvfe_mc(
                self.hparams['nll'],
                self.hparams['batch_size'] / self.hparams['noise_batch_size']
                if self.hparams['equal_weight']
                else self.hparams.get('beta', 1.0) / n_batches,
                self.hparams['prior'],
                self.model.apply,
                self.hparams['df']
            )
        ))


class LikelihoodExactGMSFSVI(GaussmixMixin, LikelihoodExactSFSVI):
    """Likelihood-focused exact-replay Gaussian-mixture S-FSVI."""

    def update_loss(self, xs, ys):
        """Update the loss function."""
        n_batches = -(len(ys) // -self.hparams['batch_size'])
        self.loss = jit(concat_fvfe(
            (gmfvfe_mc if self.hparams['mc'] else gmfvfe_ub)(
                self.hparams['nll'],
                self.hparams['batch_size'] / self.hparams['noise_batch_size']
                if self.hparams['equal_weight']
                else self.hparams.get('beta', 1.0) / n_batches,
                self.hparams['prior'],
                self.model.apply
            )
        ))
