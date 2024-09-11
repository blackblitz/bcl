"""Sequential variational inference."""

from flax.training.train_state import TrainState
from jax import jit, random, tree_util
import jax.numpy as jnp
from optax import adam

from .base import ParallelTrainer, StandardTrainer
from .init import standard_init, gvi_init, gmvi_init
from .loss import gfsvi_vfe, gmvi_vfe, gvi_vfe, sigmoid_ce, softmax_ce
from .predict import sigmoid_bma, softmax_bma
from .probability import (
    gauss_param, gauss_gumbel_sample, gauss_sample, get_gauss_prior, gsgauss_param
)
from .replay import GDumbMixin, JointMixin, NoiseMixin


class GVCL(GaussianMixin, StandardTrainer):
    """Gaussian variational continual learning."""

    def precompute(self):
        """Precompute."""
        keys = self._make_keys(
            ['precompute', 'init_state', 'update_state']
        )
        key1, key2 = random.split(keys['keys']['precompute'])
        params = standard_init(key1, self.model, self.metadata['input_shape'])
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
                self._choose(sigmoid_ce, softmax_ce)(0.0, self.model.apply),
                self.precomputed['sample'],
                self.mutables['prior'],
                self.immutables.get('beta', 1 / n_batches)
            )
        )

    def update_mutables(self, xs, ys):
        """Update the hyperparameters."""
        self.mutables['prior'] = self.state.params


class GMVCL(GaussianMixtureMixin, StandardTrainer):
    """Gaussian-mixture variational continual learning."""

    def precompute(self):
        """Precompute."""
        keys = self._make_keys(
            ['precompute', 'init_state', 'update_state']
        )
        key1, key2 = random.split(keys['keys']['precompute'])
        params = standard_init(key1, self.model, self.metadata['input_shape'])
        sample = {
            'sample': gauss_gumbel_sample(
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
        self.loss = gmvi_vfe(
            self._choose(sigmoid_ce, softmax_ce)(0.0, self.model.apply),
            self.precomputed['sample'],
            self.mutables['prior'],
            self.immutables.get('beta', 1 / n_batches)
        )

    def update_mutables(self, xs, ys):
        """Update the mutable hyperparameters."""
        self.mutables['prior'] = self.state.params


class GSFSVI(GaussianMixin, NoiseMixin, JointMixin, ParallelTrainer):
    """Gaussian sequential function-space variational inference."""

    def precompute(self):
        """Precompute."""
        keys = self._make_keys([
            'precompute', 'init_state', 'init_coreset',
            'update_state', 'update_coreset'
        ])
        key1, key2 = random.split(keys['keys']['precompute'])
        params = standard_init(key1, self.model, self.metadata['input_shape'])
        sample = {
            'sample': gauss_sample(
                key2, self.immutables['sample_size'], params
            )
        }
        return super().precompute() | keys | sample

    def init_mutables(self):
        """Initialize the mutable hyperparameters."""
        return {
            'coreset': self.init_coreset(),
            'prior': get_gauss_prior(
                self.immutables['precision'], self.state.params
            )
        }

    def update_loss(self, xs, ys):
        """Update the loss function."""
        n_batches = -(len(ys) // -self.immutables['batch_size'])
        self.loss = jit(
            gfsvi_vfe(
                self._choose(sigmoid_ce, softmax_ce)(0.0, self.model.apply),
                self.precomputed['sample'],
                self.mutables['prior'],
                self.immutables.get('beta', 1 / n_batches),
                self.model.apply
            )
        )

    def update_mutables(self, xs, ys):
        """Update the hyperparameters."""
        self.mutables['prior'] = self.state.params
        #self.update_coreset(xs, ys)
