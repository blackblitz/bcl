"""Sequential variational inference."""

from flax.training.train_state import TrainState
from jax import jit, random, tree_util
import jax.numpy as jnp
from optax import adam

from .base import StandardTrainer
from .init import standard_init, gvi_init, gmvi_init
from .loss import gmvi_vfe, gvi_vfe, sigmoid_ce, softmax_ce
from .predict import sigmoid_bma, softmax_bma
from .probability import (
    gauss_param, gauss_gumbel_sample, gauss_sample, get_gauss_prior, gsgauss_param
)


class GaussianMixin:
    """Mixin for Gaussian variation inference."""

    def init_state(self):
        """Initialize the state."""
        return TrainState.create(
            apply_fn=self.model.apply,
            params=gvi_init(
                self.precomputed['keys']['init_state'],
                self.model, self.metadata['input_shape']
            ),
            tx=adam(self.immutables['lr'])
        )

    def make_predict(self):
        """Make a predicting function."""
        return self._choose(sigmoid_bma, softmax_bma)(
            self.model.apply,
            gauss_param(self.state.params, self.precomputed['sample'])
        )


class GaussianMixtureMixin:
    """Mixin for Gaussian-mixture variation inference."""

    def init_state(self):
        """Initialize the state."""
        return TrainState.create(
            apply_fn=self.model.apply,
            params=gmvi_init(
                self.precomputed['keys']['init_state'],
                self.immutables['n_comp'],
                self.model,
                self.metadata['input_shape']
            ),
            tx=adam(self.immutables['lr'])
        )

    def make_predict(self):
        """Make a predicting function."""
        return self._choose(sigmoid_bma, softmax_bma)(
            self.model.apply,
            gsgauss_param(self.state.params, self.precomputed['sample'])
        )


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
