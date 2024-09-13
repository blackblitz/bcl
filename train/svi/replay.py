"""Replay."""

from jax import jit, random
import jax.numpy as jnp

from ..coreset import TaskIncrementalCoreset
from ..loss import basic_loss, gmfsvi_vfe, gfsvi_vfe
from ..probability import gsgauss_sample, gauss_sample, get_gauss_prior
from ..state.functions import init
from ..state.mixins import GSGaussMixin, GaussMixin, ParallelChoiceMixin
from ..trainer import ContinualTrainer


class GSFSVI(GaussMixin, ParallelChoiceMixin, ContinualTrainer):
    """Gaussian sequential function-space variational inference."""

    def precompute(self):
        """Precompute."""
        keys = self._make_keys([
            'precompute', 'init_state', 'init_coreset',
            'update_state', 'update_coreset'
        ])
        key1, key2 = random.split(keys['keys']['precompute'])
        params = init(key1, self.model, self.metadata['input_shape'])
        sample = {
            'sample': gauss_sample(
                key2, self.immutables['sample_size'], params
            )
        }
        return super().precompute() | keys | sample

    def init_mutables(self):
        """Initialize the mutable hyperparameters."""
        return {
            'coreset': TaskIncrementalCoreset(
                'coreset.zarr', 'coreset.memmap',
                self.immutables, self.metadata
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
                    self.immutables['nntype'],
                    0.0,
                    self.model.apply
                ),
                self.precomputed['sample'],
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


class GMSFSVI(GSGaussMixin, ParallelChoiceMixin, ContinualTrainer):
    """Gaussian sequential function-space variational inference."""

    def precompute(self):
        """Precompute."""
        keys = self._make_keys([
            'precompute', 'init_state', 'init_coreset',
            'update_state', 'update_coreset'
        ])
        key1, key2 = random.split(keys['keys']['precompute'])
        params = init(key1, self.model, self.metadata['input_shape'])
        sample = {
            'sample': gsgauss_sample(
                key2, self.immutables['sample_size'],
                self.immutables['n_comp'], params
            )
        }
        return super().precompute() | keys | sample

    def init_mutables(self):
        """Initialize the mutable hyperparameters."""
        return {
            'coreset': TaskIncrementalCoreset(
                'coreset.zarr', 'coreset.memmap',
                self.immutables, self.metadata
            ),
            'prior': get_gauss_prior(
                self.immutables['precision'], self.state.params
            ) | {'logit': jnp.zeros_like(self.state.params['logit'])}
        }

    def update_loss(self, xs, ys):
        """Update the loss function."""
        n_batches = -(len(ys) // -self.immutables['batch_size'])
        self.loss = jit(
            gmfsvi_vfe(
                basic_loss(
                    self.immutables['nntype'],
                    0.0,
                    self.model.apply
                ),
                self.precomputed['sample'],
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
