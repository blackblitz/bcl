"""Mixins for the training state."""

from itertools import cycle

from flax.training.train_state import TrainState
from jax import random
from optax import sgd

from dataops.array import batch, shuffle

from .functions import init, gauss_init, gsgauss_init, make_step
from ..predict import MAPPredictor


class MAPMixin:
    """Mixin for MAP inference."""

    predictor = MAPPredictor

    def init_state(self):
        """Initialize the state."""
        return TrainState.create(
            apply_fn=self.model.apply,
            params=init(
                self.precomputed['keys']['init_state'],
                self.model, self.metadata['input_shape']
            ),
            tx=sgd(self.immutables['lr'])
        )


class GaussMixin:
    """Mixin for Gaussian variation inference."""

    def init_state(self):
        """Initialize the state."""
        return TrainState.create(
            apply_fn=self.model.apply,
            params=gauss_init(
                self.precomputed['keys']['init_state'],
                self.model, self.metadata['input_shape']
            ),
            tx=sgd(self.immutables['lr'])
        )


class GSGaussMixin:
    """Mixin for Gaussian-mixture variation inference."""

    def init_state(self):
        """Initialize the state."""
        return TrainState.create(
            apply_fn=self.model.apply,
            params=gsgauss_init(
                self.precomputed['keys']['init_state'],
                self.model, self.metadata['input_shape']
            ),
            tx=sgd(self.immutables['lr'])
        )


class RegularMixin:
    """Mixin for regular SGD."""

    def update_state(self, xs, ys):
        """Update the training state."""
        step = make_step(self.loss)
        for key in random.split(
            self.precomputed['keys']['update_state'],
            num=self.immutables['n_epochs']
        ):
            for indices in batch(
                self.immutables['batch_size'], shuffle(key, len(ys))
            ):
                self.state = step(self.state, xs[indices], ys[indices])


class SerialMixin:
    """Mixin for SGD with coreset in series."""

    def update_state(self, xs, ys):
        """Update the training state."""
        self.mutables['coreset'].create_memmap()
        step = make_step(self.loss)
        for key in random.split(
            self.precomputed['keys']['update_state'],
            num=self.immutables['n_epochs']
        ):
            key1, key2 = random.split(key)
            for indices in batch(
                self.immutables['batch_size'], shuffle(key1, len(ys))
            ):
                self.state = step(self.state, xs[indices], ys[indices])
            for xs_batch, ys_batch in (
                self.mutables['coreset'].shuffle_batch(key2)
            ):
                self.state = step(self.state, xs_batch, ys_batch)
        self.mutables['coreset'].delete_memmap()


class ParallelShuffleMixin:
    """Mixin for SGD with coreset in parallel by shuffle-batching."""

    def update_state(self, xs, ys):
        """Update the training state."""
        self.mutables['coreset'].create_memmap()
        step = make_step(self.loss)
        for key in random.split(
            self.precomputed['keys']['update_state'],
            num=self.immutables['n_epochs']
        ):
            key1, key2 = random.split(key)
            for (indices, (xs_batch, ys_batch)) in zip(
                batch(
                    self.immutables['batch_size'], shuffle(key1, len(ys))
                ),
                cycle(self.mutables['coreset'].shuffle_batch(key2))
            ):
                self.state = step(
                    self.state, xs[indices], ys[indices], xs_batch, ys_batch
                )
        self.mutables['coreset'].delete_memmap()


class ParallelChoiceMixin:
    """Mixin for SGD with coreset in parallel by choice."""

    def update_state(self, xs, ys):
        """Update the training state."""
        self.mutables['coreset'].create_memmap()
        step = make_step(self.loss)
        for key in random.split(
            self.precomputed['keys']['update_state'],
            num=self.immutables['n_epochs']
        ):
            keys = random.split(
                key, num=-(len(ys) // -self.immutables['batch_size']) + 1
            )
            for i, indices in enumerate(
                batch(
                    self.immutables['batch_size'],
                    shuffle(keys[0], len(ys))
                ),
                start=1
            ):
                self.state = step(
                    self.state, xs[indices], ys[indices],
                    *self.mutables['coreset'].choice(keys[i])
                )
        self.mutables['coreset'].delete_memmap()
