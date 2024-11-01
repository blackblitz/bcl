"""Replay."""

from jax import jit, random
import jax.numpy as jnp

from dataops.array import batch, get_n_batches, shuffle

from ..base import ContinualTrainer, GaussMixin, GSGaussMixin, TMixin
from ..coreset import TaskIncrementalCoreset
from ...training.loss.stateless import (
    gpvfe_cf, gpvfe_mc, gfvfe_cf, gfvfe_mc, tpvfe_mc, tfvfe_mc,
    gmpvfe_ub, gmpvfe_mc, gmfvfe_ub, gmfvfe_mc,
)
from ...training.stateless import make_step
from ...training.vi import gauss, gaussmix, t


class PriorExactSFSVI(ContinualTrainer):
    """Prior-focused exact-replay S-FSVI."""

    def update_state(self, xs, ys):
        """Update the training state."""
        self.mutables['coreset'].create_memmap()
        step = make_step(self.loss)
        for epoch_key in random.split(
            self.precomputed['keys']['update_state'],
            num=self.immutables['n_epochs']
        ):
            n_batches = get_n_batches(len(ys), self.immutables['batch_size'])
            epoch_key, shuffle_key = random.split(epoch_key)
            for key, indices in zip(
                random.split(epoch_key, num=n_batches),
                batch(
                    self.immutables['batch_size'],
                    shuffle(shuffle_key, len(ys))
                )
            ):
                key1, key2, key3, key4 = random.split(key, num=4)
                param_sample = self.sample(key1)
                if (
                    self.mutables['coreset'].task_count > 0
                    and random.bernoulli(
                        key2,
                        p=self.immutables['coreset_prob']
                    )
                ):
                    ind_xs, _ = self.mutables['coreset'].choice(
                        key3,
                        self.immutables['coreset_batch_size_per_task']
                    )
                else:
                    ind_xs = random.uniform(
                        key3,
                        shape=(
                            self.immutables['coreset_batch_size_per_task'],
                            *self.model_spec.in_shape
                        ),
                        minval=jnp.array(self.immutables['noise_minval']),
                        maxval=jnp.array(self.immutables['noise_maxval']),
                    )
                output_sample = self.sample(key4, xs=ind_xs)
                self.state = step(
                    self.state, param_sample, output_sample,
                    xs[indices], ys[indices], ind_xs
                )
            yield self.state
        self.mutables['coreset'].delete_memmap()

    def update_mutables(self, xs, ys):
        """Update the hyperparameters."""
        self.mutables['prior'] = self.state.params
        self.mutables['coreset'].update(
            self.precomputed['keys']['update_mutables'], xs, ys
        )


class PriorExactGSFSVI(GaussMixin, PriorExactSFSVI):
    """Prior-focused exact-replay Gaussian S-FSVI."""

    def init_mutables(self):
        """Initialize the mutable hyperparameters."""
        return {
            'coreset': TaskIncrementalCoreset(
                'coreset.zarr', 'coreset.memmap',
                self.model_spec, self.immutables['coreset_size_per_task']
            ),
            'prior': gauss.get_prior(
                self.immutables['precision'], self.state.params
            )
        }

    def update_loss(self, xs, ys):
        """Update the loss function."""
        n_batches = -(len(ys) // -self.immutables['batch_size'])
        self.loss = jit(
            (gfvfe_mc if self.immutables['mc'] else gfvfe_cf)(
                self.precomputed['nll'],
                self.immutables.get('beta', 1 / n_batches),
                self.mutables['prior'],
                self.model.apply
            )
        )


class PriorExactTSFSVI(TMixin, PriorExactSFSVI):
    """Prior-focused exact-replay Student's t S-FSVI."""

    def init_mutables(self):
        """Initialize the mutable hyperparameters."""
        return {
            'coreset': TaskIncrementalCoreset(
                'coreset.zarr', 'coreset.memmap',
                self.model_spec, self.immutables['coreset_size_per_task']
            ),
            'prior': t.get_prior(
                self.immutables['invscale'], self.state.params
            )
        }

    def update_loss(self, xs, ys):
        """Update the loss function."""
        n_batches = -(len(ys) // -self.immutables['batch_size'])
        self.loss = jit(
            tfvfe_mc(
                self.precomputed['nll'],
                self.immutables.get('beta', 1 / n_batches),
                self.mutables['prior'],
                self.model.apply,
                self.immutables['df']
            )
        )


class PriorExactGMSFSVI(GSGaussMixin, PriorExactSFSVI):
    """Prior-focused exact-replay Gaussian-mixture S-FSVI."""

    def init_mutables(self):
        """Initialize the mutable hyperparameters."""
        return {
            'coreset': TaskIncrementalCoreset(
                'coreset.zarr', 'coreset.memmap',
                self.model_spec, self.immutables['coreset_size_per_task']
            ),
            'prior': gaussmix.get_prior(
                self.immutables['precision'], self.state.params
            )
        }

    def update_loss(self, xs, ys):
        """Update the loss function."""
        n_batches = -(len(ys) // -self.immutables['batch_size'])
        self.loss = jit(
            (gmfvfe_mc if self.immutables['mc'] else gmfvfe_ub)(
                self.precomputed['nll'],
                self.immutables.get('beta', 1 / n_batches),
                self.mutables['prior'],
                self.model.apply
            )
        )
