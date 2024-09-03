"""Predicting functions."""

from jax import vmap
from jax.nn import sigmoid, softmax


def sigmoid_map(apply, params):
    """Return a MAP prediction function for a sigmoid-activated function."""
    def predict(xs, decide=True):
        proba = sigmoid(apply({'params': params}, xs)[:, 0])
        return proba >= 0.5 if decide else proba

    return predict


def softmax_map(apply, params):
    """Return a MAP prediction function for a softmax-activation function."""
    def predict(xs, decide=True):
        proba = softmax(apply({'params': params}, xs))
        return proba.argmax(axis=-1) if decide else proba

    return predict


def sigmoid_bma(apply, paramss):
    """Return a BMA prediction function for a sigmoid-activated function."""
    def predict(xs, decide=True):
        proba = vmap(
            lambda params: sigmoid(apply({'params': params}, xs)[:, 0])
        )(paramss)
        return proba >= 0.5 if decide else proba

    return predict


def softmax_bma(apply, paramss):
    """Return a BMA prediction function for a softmax-activated function."""
    def predict(xs, decide=True):
        proba = vmap(
            lambda params: softmax(apply({'params': params}, xs))
        )(paramss)
        return proba.argmax(axis=-1) if decide else proba

    return predict
