"""Simple trainers."""

from jax import jit, random
import jax.numpy as jnp

from dataops.array import batch, get_n_batches, shuffle

from ..loss.stateless import (
    gpvfe_cf, gpvfe_mc, gfvfe_cf, gfvfe_mc, tpvfe_mc, tfvfe_mc,
    gmpvfe_ub, gmpvfe_mc, gmfvfe_ub, gmfvfe_mc,
)
from ..probability import get_gauss_prior, get_t_prior
from ..trainer import ContinualTrainer, GaussMixin, GSGaussMixin, TMixin
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
                    self.state, self.sample(key3),
                    xs[indices], ys[indices]
                )
            yield self.state

    def update_mutables(self, xs, ys):
        """Update the hyperparameters."""
        self.mutables['prior'] = self.state.params


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
            (gpvfe_mc if self.immutables['mc'] else gpvfe_cf)(
                self.precomputed['nll'],
                self.immutables.get('beta', 1 / n_batches),
                self.mutables['prior']
            )
        )


class TVCL(TMixin, VCL):
    """t variational continual learning."""

    def init_mutables(self):
        """Initialize the mutable hyperparameters."""
        return {
            'prior': get_t_prior(
                self.immutables['invscale'], self.state.params
            )
        }

    def update_loss(self, xs, ys):
        """Update the loss function."""
        n_batches = -(len(ys) // -self.immutables['batch_size'])
        self.loss = jit(
            tpvfe_mc(
                self.precomputed['nll'],
                self.immutables.get('beta', 1 / n_batches),
                self.mutables['prior'],
                self.immutables['df']
            )
        )


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
            (gmpvfe_mc if self.immutables['mc'] else gmpvfe_ub)(
                self.precomputed['nll'],
                self.immutables.get('beta', 1 / n_batches),
                self.mutables['prior']
            )
        )


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
                ind_xs = random.uniform(
                    key2,
                    shape=(
                        self.immutables['noise_batch_size'],
                        *self.model_spec.in_shape
                    ),
                    minval=jnp.array(self.immutables['noise_minval']),
                    maxval=jnp.array(self.immutables['noise_maxval']),
                )
                self.state = step(
                    self.state,
                    self.sample(key1), self.sample(key3, xs=ind_xs),
                    xs[indices], ys[indices], ind_xs
                )
            yield self.state

    def update_mutables(self, xs, ys):
        """Update the hyperparameters."""
        self.mutables['prior'] = self.state.params


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
            (gfvfe_mc if self.immutables['mc'] else gfvfe_cf)(
                self.precomputed['nll'],
                self.immutables.get('beta', 1 / n_batches),
                self.mutables['prior'],
                self.model.apply
            )
        )


class SimpleTSFSVI(TMixin, SimpleSFSVI):
    """Simple t S-FSVI."""

    def init_mutables(self):
        """Initialize the mutable hyperparameters."""
        return {
            'prior': get_t_prior(
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
            (gmfvfe_mc if self.immutables['mc'] else gmfvfe_ub)(
                self.precomputed['nll'],
                self.immutables.get('beta', 1 / n_batches),
                self.mutables['prior'],
                self.model.apply
            )
        )
