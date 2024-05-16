"""Training package."""

from optax import (
    sigmoid_binary_cross_entropy, softmax_cross_entropy_with_integer_labels
)

from dataio import iter_batches

from .loss import make_loss, make_loss_reg, make_step


class Trainer:
    """Trainer for continual learning."""

    def __init__(
        self, state, hyperparams,
        batch_size_hyperparams=1024, batch_size_state=64,
        multiclass=True, n_epochs=10
    ):
        """Intialize self."""
        self.state = state
        self.hyperparams = hyperparams
        self.batch_size_hyperparams = batch_size_hyperparams
        self.batch_size_state = batch_size_state
        self.multiclass = multiclass
        self.n_epochs = n_epochs
        if multiclass:
            self.loss_basic = make_loss(
                self.state,
                softmax_cross_entropy_with_integer_labels,
                multi=True
            )
        else:
            self.loss_basic = make_loss(
                self.state,
                sigmoid_binary_cross_entropy,
                multi=False
            )
        self.loss = None
        self.n_obs = 0

    def train(self, x, y):
        """Train self."""
        self.update_loss(x, y)
        self.update_state(x, y)
        self.update_hyperparams(x, y)
        self.n_obs += len(y)

    def update_loss(self, x, y):  # pylint: disable=unused-argument
        """Update loss function."""
        self.loss = make_loss_reg(
            self.hyperparams['precision'], self.loss_basic
        )

    def update_state(self, x, y):
        """Update state."""
        step = make_step(self.loss)
        for x_batch, y_batch in iter_batches(
            self.n_epochs, self.batch_size_state, x, y
        ):
            self.state = step(self.state, x_batch, y_batch)

    def update_hyperparams(self, x, y):
        """Update hyperparameters."""
