"""Regular training."""

from . import Trainer
from .loss import make_loss_reg


class RegularTrainer(Trainer):
    """Regular trainer."""

    def update_loss(self, x, y):
        """Update loss function."""
        self.loss = make_loss_reg(
            self.hyperparams['precision'], self.loss_basic
        )

    def update_hyperparams(self, x, y):
        """Update hyperparameters."""
