"""Replay."""

from jax import jit, random
import jax.numpy as jnp

from dataops.array import batch, get_n_batches, shuffle

from ..coreset import TaskIncrementalCoreset
from ..loss import gmfsvi_vfe_mc, gmfsvi_vfe_ub, gfsvi_vfe
from ..probability import get_gauss_prior
from ..state.functions import make_step
from ..state.mixins import GSGaussMixin, GaussMixin
from ..trainer import ContinualTrainer


class SFSVI(ContinualTrainer):
    """Sequential function-space variational inference."""

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
                key1, key2, key3 = random.split(key, num=3)
                self.state = step(
                    self.state, self.sample(key1), xs[indices], ys[indices],
                    *(
                        self.mutables['coreset'].choice(
                            key3,
                            self.immutables['coreset_batch_size_per_task']
                        ) if (
                            self.mutables['coreset'].task_count > 0
                            and random.bernoulli(
                                key2,
                                p=self.immutables['coreset_prob']
                            )
                        ) else self.mutables['coreset'].noise(
                            key3,
                            self.immutables['coreset_batch_size_per_task'],
                            minval=self.immutables['noise_minval'],
                            maxval=self.immutables['noise_maxval']
                        )
                    )
                )
            yield self.state
        self.mutables['coreset'].delete_memmap()


class GSFSVI(GaussMixin, SFSVI):
    """Gaussian sequential function-space variational inference."""

    def init_mutables(self):
        """Initialize the mutable hyperparameters."""
        return {
            'coreset': TaskIncrementalCoreset(
                'coreset.zarr', 'coreset.memmap',
                self.model_spec, self.immutables['coreset_size_per_task']
            ),
            'prior': get_gauss_prior(
                self.immutables['precision'], self.state.params
            )
        }

    def update_loss(self, xs, ys):
        """Update the loss function."""
        n_batches = -(len(ys) // -self.immutables['batch_size'])
        self.loss = jit(
            gfsvi_vfe(
                self.precomputed['nll'],
                self.mutables['prior'],
                self.immutables.get('beta', 1 / n_batches),
                self.model.apply
            )
        )

    def update_mutables(self, xs, ys):
        """Update the hyperparameters."""
        self.mutables['prior'] = self.state.params
        self.mutables['coreset'].update(
            self.precomputed['keys']['update_coreset'], xs, ys
        )


class GMSFSVI(GSGaussMixin, SFSVI):
    """Gaussian sequential function-space variational inference."""

    def init_mutables(self):
        """Initialize the mutable hyperparameters."""
        return {
            'coreset': TaskIncrementalCoreset(
                'coreset.zarr', 'coreset.memmap',
                self.model_spec, self.immutables['coreset_size_per_task']
            ),
            'prior': get_gauss_prior(
                self.immutables['precision'], self.state.params
            ) | {'logit': jnp.zeros_like(self.state.params['logit'])}
        }

    def update_loss(self, xs, ys):
        """Update the loss function."""
        n_batches = -(len(ys) // -self.immutables['batch_size'])
        self.loss = jit(
            (gmfsvi_vfe_mc if self.immutables['mc_kldiv'] else gmfsvi_vfe_ub)(
                *((
                    [self.precomputed['keys']['update_loss']]
                    if self.immutables['mc_kldiv'] else []
                ) + [
                    self.precomputed['nll'],
                    self.mutables['prior'],
                    self.immutables.get('beta', 1 / n_batches),
                    self.model.apply
                ])
            )
        )

    def update_mutables(self, xs, ys):
        """Update the hyperparameters."""
        self.mutables['prior'] = self.state.params
        self.mutables['coreset'].update(
            self.precomputed['keys']['update_mutables'], xs, ys
        )
