"""Neural Consolidation."""

from jax import flatten_util, jit, random, vmap
from optax import huber_loss

from dataio import iter_batches

from .loss import make_loss_reg, make_loss, make_step

from . import Trainer


class NeuralConsolidation(Trainer):
    """Neural Consolidation."""

    def update_loss(self, x, y):
        """Update loss function."""
        if self.n_obs > 0:
            self.loss = jit(
                lambda params, x, y:
                self.hyperparams['state'].apply_fn(
                    {'params': self.hyperparams['state'].params},
                    flatten_util.ravel_pytree(params)[0]
                )[0] + self.loss_basic(params, x, y)
            )
        else:
            self.loss = make_loss_reg(
                self.hyperparams['precision'], self.loss_basic
            )

    def _make_data(self, key, x, y):
        """Make data for neural consolidation."""
        pflat, punflatten = flatten_util.ravel_pytree(
            self.hyperparams['minimum']
        )
        sample = random.ball(
            key, len(pflat), shape=(self.hyperparams['size'],)
        )
        pflats = pflat + self.hyperparams['radius'] * sample
        losses = sum(
            vmap(
                self.loss, in_axes=(0, None, None)
            )(vmap(punflatten)(pflats), x_batch, y_batch)
            for x_batch, y_batch in iter_batches(
                1, self.batch_size_hyperparams, x, y, shuffle=False
            )
        )
        return pflats, losses

    def update_hyperparams(self, x, y):
        """Update hyperparameters."""
        self.hyperparams['minimum'] = self.state.params
        state = self.hyperparams['state']
        loss = make_loss_reg(
            self.hyperparams['reg'], make_loss(state, huber_loss, multi=False)
        )
        step = make_step(loss)
        for key in random.split(
            random.PRNGKey(1337), num=self.hyperparams['nsteps']
        ):
            pflats, losses = self._make_data(key, x, y)
            state = step(state, pflats, losses)
        self.hyperparams['state'] = state
