"""Replay."""

from jax import jit, random
import jax.numpy as jnp

from dataops.array import batch, get_n_batches, shuffle

from ..coreset import TaskIncrementalCoreset
from ..loss import basic_loss, gmfsvi_vfe_mc, gmfsvi_vfe_ub, gfsvi_vfe
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
        for key in random.split(
            self.precomputed['keys']['update_state'],
            num=self.immutables['n_epochs']
        ):
            key1, key2, key3 = random.split(key, num=3)
            n_batches = get_n_batches(len(ys), self.immutables['batch_size'])
            for key3, key4, indices in zip(
                random.split(key1, num=n_batches),
                random.split(key2, num=n_batches),
                batch(
                    self.immutables['batch_size'],
                    shuffle(key1, len(ys))
                )
            ):
                self.state = step(
                    self.state, self.sample(key3), xs[indices], ys[indices],
                    *(
                        self.mutables['coreset'].choice(
                            key4, self.immutables['coreset_batch_size_per_task']
                        ) if self.mutables['coreset'].task_count > 0
                        else self.mutables['coreset'].noise(
                            key4, self.immutables['coreset_batch_size_per_task'],
                            minval=self.immutables['noise_minval'],
                            maxval=self.immutables['noise_maxval']
                        )
                    )
                )
            yield self.state
        self.mutables['coreset'].delete_memmap()


class GSFSVI(GaussMixin, SFSVI):
    """Gaussian sequential function-space variational inference."""

    def precompute(self):
        """Precompute."""
        return super().precompute() | self._make_keys([
            'init_state', 'init_coreset', 'update_state', 'update_coreset'
        ])

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
                basic_loss(
                    self.model_spec.fin_act,
                    0.0,
                    self.model.apply
                ),
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

    def precompute(self):
        """Precompute."""
        if self.immutables['mc_kldiv']:
            return super().precompute() | self._make_keys([
                'init_state', 'init_coreset', 'update_loss',
                'update_state', 'update_coreset'
            ])
        return super().precompute() | self._make_keys([
            'init_state', 'init_coreset', 'update_state', 'update_coreset'
        ])

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
                    basic_loss(
                        self.model_spec.fin_act,
                        0.0,
                        self.model.apply
                    ),
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
            self.precomputed['keys']['update_coreset'], xs, ys
        )
