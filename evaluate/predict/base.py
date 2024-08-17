"""Base classes."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

from jax.nn import sigmoid, softmax


@dataclass
class Predictor(ABC):
    """Abstract class for a predictor."""

    apply: Callable

    @staticmethod
    @abstractmethod
    def _apply(apply, params, xs):
        """Apply the model."""

    @staticmethod
    @abstractmethod
    def _activate(fs):
        """Activate the output of the model."""

    @staticmethod
    @abstractmethod
    def _decide(ps):
        """Decide a class based on class probabilities."""

    @abstractmethod
    def predict_proba(self, xs):
        """Predict class probabilities."""

    @abstractmethod
    def predict(self, xs):
        """Predict a class."""


class SigmoidMixin:
    """Mixin for sigmoid classification."""

    @staticmethod
    def _apply(apply, params, xs):
        """Predict class probabilities."""
        return apply({'params': params}, xs)[:, 0]

    @staticmethod
    def _activate(fs):
        """Activate the output of the model."""
        return sigmoid(fs)

    @staticmethod
    def _decide(ps):
        """Decide a class based on class probabilities."""
        return ps >= 0.5


class SoftmaxMixin:
    """Mixin for softmax classification."""

    @staticmethod
    def _apply(apply, params, xs):
        """Predict class probabilities."""
        return apply({'params': params}, xs)

    @staticmethod
    def _activate(fs):
        """Activate the output of the model."""
        return softmax(fs)

    @staticmethod
    def _decide(ps):
        """Decide a class based on class probabilities."""
        return ps.argmax(axis=-1)
