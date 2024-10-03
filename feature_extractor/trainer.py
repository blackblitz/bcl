"""Trainer."""

from flax.training.train_state import TrainState
from jax import random
import jax.numpy as jnp
import optax

from dataops.array import batch, shuffle
from train.loss.stateful import get_nll, l2_reg
from train.training.stateful import make_step


class Trainer:
    """Trainer for feature extractor."""

    def __init__(self, model, model_spec, hyperparams):
        """Initialize self."""
        self.model = model
        self.model_spec = model_spec
        self.hyperparams = hyperparams
        init_key, dropout_key, shuffle_key = random.split(
            random.key(self.hyperparams['seed']), num=3
        )
        self.keys = {'dropout': dropout_key, 'shuffle': shuffle_key}
        var = self.model.init(
            init_key, jnp.zeros((1, *self.model_spec.in_shape)),
            train=False
        )
        self.state = TrainState.create(
            apply_fn=self.model.apply,
            params=var.pop('params'),
            tx=optax.adam(self.hyperparams['lr'])
        )
        self.var = var
        self.loss = l2_reg(
            self.hyperparams['precision'],
            get_nll(self.model_spec.nll)(
                self.model.apply, mutable=list(self.var.keys()),
                train=True
            )
        )

    def train(self, xs, ys):
        """Train with a dataset."""
        step = make_step(self.keys['dropout'], self.loss)
        keys = random.split(
            self.keys['shuffle'],
            num=self.hyperparams['n_epochs']
        )
        for key in keys:
            for indices in batch(
                self.hyperparams['batch_size'], shuffle(key, len(ys))
            ):
                self.state, self.var = step(
                    self.state, self.var, xs[indices], ys[indices]
                )
            yield self.state, self.var
