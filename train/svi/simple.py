"""Simple trainers."""

from jax import jit, random
import jax.numpy as jnp

from ..loss import basic_loss, gmvi_vfe_mc, gmvi_vfe_ub, gvi_vfe
from ..probability import gsgauss_sample, gauss_sample, get_gauss_prior
from ..state.functions import init
from ..state.mixins import GaussMixin, GSGaussMixin, RegularMixin
from ..trainer import ContinualTrainer


class GVCL(GaussMixin, RegularMixin, ContinualTrainer):
    """Gaussian variational continual learning."""

    def precompute(self):
        """Precompute."""
        keys = self._make_keys(
            ['precompute', 'init_state', 'update_state']
        )
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
            'prior': get_gauss_prior(
                self.immutables['precision'], self.state.params
            )
        }

    def update_loss(self, xs, ys):
        """Update the loss function."""
        n_batches = -(len(ys) // -self.immutables['batch_size'])
        self.loss = jit(
            gvi_vfe(
                basic_loss(
                    self.immutables['nntype'],
                    0.0,
                    self.model.apply
                ),
                self.precomputed['sample'],
                self.mutables['prior'],
                self.immutables.get('beta', 1 / n_batches)
            )
        )

    def update_mutables(self, xs, ys):
        """Update the hyperparameters."""
        self.mutables['prior'] = self.state.params


class GMVCL(GSGaussMixin, RegularMixin, ContinualTrainer):
    """Gaussian-mixture variational continual learning."""

    def precompute(self):
        """Precompute."""
        if self.immutables['mc_kldiv']:
            keys = self._make_keys(
                ['precompute', 'init_state', 'update_loss', 'update_state']
            )
        else:
            keys = self._make_keys(
                ['precompute', 'init_state', 'update_state']
            )
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
            'prior': get_gauss_prior(
                self.immutables['precision'], self.state.params
            ) | {'logit': jnp.zeros_like(self.state.params['logit'])}
        }

    def update_loss(self, xs, ys):
        """Update the loss function."""
        n_batches = -(len(ys) // -self.immutables['batch_size'])
        self.loss = jit(
            (gmvi_vfe_mc if self.immutables['mc_kldiv'] else gmvi_vfe_ub)(
                basic_loss(
                    self.immutables['nntype'],
                    0.0,
                    self.model.apply
                ),
                self.precomputed['sample'],
                self.mutables['prior'],
                self.immutables.get('beta', 1 / n_batches),
            )
        )

    def update_mutables(self, xs, ys):
        """Update the mutable hyperparameters."""
        self.mutables['prior'] = self.state.params
