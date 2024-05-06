"""Training package."""

from abc import ABC, abstractmethod

from jax import random, tree_util

from dataio import iter_batches

from .loss import make_step


def normaltree(key, n, tree):
    keys = random.split(key, len(tree_util.tree_leaves(tree)))
    keys = tree_util.tree_unflatten(tree_util.tree_structure(tree), keys)
    return tree_util.tree_map(
        lambda x, key: random.normal(key, (n, *x.shape)), tree, keys
    )


class Trainer(ABC):
    def __init__(self, state, hyperparams, loss_basic):
        self.state = state
        self.hyperparams = hyperparams
        self.loss_basic = loss_basic

    def train(self, n_epochs, minibatch_size, batch_size, x, y):
        self.update_loss()
        self.update_state(n_epochs, minibatch_size, x, y)
        self.update_hyperparams(batch_size, x, y)

    @abstractmethod
    def update_loss(self):
        pass

    def update_state(self, n_epochs, minibatch_size, x, y):
        step = make_step(self.loss)
        for minibatch_x, minibatch_y in iter_batches(n_epochs, minibatch_size, x, y):
            self.state = step(self.state, minibatch_x, minibatch_y)

    @abstractmethod
    def update_hyperparams(self, batch_size, x, y):
        pass
