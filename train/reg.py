"""Regular training."""

from operator import add

from jax import jit, tree_util

from . import Trainer
from .loss import make_loss_reg


class RegularTrainer(Trainer):
    def update_loss(self):
        self.loss = make_loss_reg(
            self.state, self.hyperparams['precision'], self.loss_basic
        )


    def update_hyperparams(self, batch_size, x, y):
        pass
