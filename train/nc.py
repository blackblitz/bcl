"""Neural Consolidation."""

from jax import flatten_util, jit, random, vmap
import numpy as np
from optax import huber_loss
from tqdm import tqdm

from dataio import iter_batches

from .loss import make_loss_reg, make_loss_single_output, make_step

from . import Trainer


class NeuralConsolidation(Trainer):
    def update_loss(self):
        if hasattr(self, 'loss'):
            self.loss = jit(
                lambda params, x, y:
                self.hyperparams['state_consolidator'].apply_fn(
                    {'params': self.hyperparams['state_consolidator'].params},
                    flatten_util.ravel_pytree(params)[0]
                )[0] + self.loss_basic(params, x, y)
            )
        else:
            self.loss = make_loss_reg(
                self.state, self.hyperparams['precision'], self.loss_basic
            )


    def _make_data(self, key, batch_size, x, y):
        pflat, punflatten = flatten_util.ravel_pytree(self.hyperparams['minimum'])
        sample = random.ball(key, len(pflat), shape=(self.hyperparams['size'],))
        pflats = pflat + self.hyperparams['scale'] * sample
        losses = sum(
            vmap(
                self.loss, in_axes=(0, None, None)
            )(vmap(punflatten)(pflats), x_batch, y_batch)
            for x_batch, y_batch in iter_batches(1, batch_size, x, y)
        )
        return pflats, losses


    def update_hyperparams(self, batch_size, x, y):
        self.hyperparams['minimum'] = self.state.params
        state = self.hyperparams['state_consolidator_init']
        loss = make_loss_reg(
            state, 1.0, make_loss_single_output(state, huber_loss)
        )
        step = make_step(loss)
        for key in random.split(
            random.PRNGKey(1337), num=self.hyperparams['nsteps']
        ):
            state = step(state, *self._make_data(key, batch_size, x, y))
        self.hyperparams['state_consolidator'] = state
