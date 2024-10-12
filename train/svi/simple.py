"""Simple trainers."""

from jax import jit, random
import jax.numpy as jnp

from dataops.array import batch, get_n_batches, shuffle

from ..loss.stateless import (
    gmvi_vfe_mc, gmvi_vfe_ub, gvi_vfe,
    gmfsvi_vfe_mc, gmfsvi_vfe_ub, gfsvi_vfe
)
from ..probability import get_gauss_prior
from ..trainer import ContinualTrainer, GaussMixin, GSGaussMixin
from ..training.stateless import make_step


class VCL(ContinualTrainer):
    """Variational continual learning."""

    def update_state(self, xs, ys):
        """Update the training state."""
        step = make_step(self.loss)
        for key in random.split(
            self.precomputed['keys']['update_state'],
            num=self.immutables['n_epochs']
        ):
            key1, key2 = random.split(key)
            n_batches = get_n_batches(len(ys), self.immutables['batch_size'])
            for key3, indices in zip(
                random.split(key1, num=n_batches),
                batch(
                    self.immutables['batch_size'], shuffle(key2, len(ys))
                )
            ):
                self.state = step(
                    self.state, self.sample(key3), xs[indices], ys[indices]
                )
            yield self.state


class GVCL(GaussMixin, VCL):
    """Gaussian variational continual learning."""

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
                self.precomputed['nll'],
                self.mutables['prior'],
                self.immutables.get('beta', 1 / n_batches)
            )
        )

    def update_mutables(self, xs, ys):
        """Update the hyperparameters."""
        self.mutables['prior'] = self.state.params


class GMVCL(GSGaussMixin, VCL):
    """Gaussian-mixture variational continual learning."""

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
                self.precomputed['nll'],
                self.mutables['prior'],
                self.immutables.get('beta', 1 / n_batches),
            )
        )

    def update_mutables(self, xs, ys):
        """Update the mutable hyperparameters."""
        self.mutables['prior'] = self.state.params


class SimpleSFSVI(ContinualTrainer):
    """Simple S-FSVI."""

    def update_state(self, xs, ys):
        """Update the training state."""
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
                    random.uniform(
                        key3,
                        shape=(
                            self.immutables['noise_batch_size'],
                            *self.model_spec.in_shape
                        ),
                        minval=jnp.array(self.immutables['noise_minval']),
                        maxval=jnp.array(self.immutables['noise_maxval']),
                    ),
                    None
                )
            yield self.state


class SimpleGSFSVI(GaussMixin, SimpleSFSVI):
    """Simple Gaussian S-FSVI."""

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


class SimpleGMSFSVI(GSGaussMixin, SimpleSFSVI):
    """Simple Gaussian-mixture S-FSVI."""

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
