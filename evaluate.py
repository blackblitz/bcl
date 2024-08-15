"""Evaluation module."""

from jax import vmap
from jax.nn import sigmoid, softmax

from dataio import iter_batches


def predict_proba_sigmoid_map(apply, params, xs):
    """Predict probabilities for sigmoid with MAP prediction."""
    return sigmoid(apply({'params': params}, xs)[:, 0])


def predict_proba_sigmoid_bma(apply, params_sample, xs):
    """Predict probabilities for sigmoid with BMA prediction."""
    return vmap(
        lambda params: predict_proba_sigmoid_map(apply, params, xs)
    )(params_sample).mean(axis=0)


def predict_proba_softmax_map(apply, params, xs):
    """Predict probabilities for softmax with MAP prediction."""
    return softmax(apply({'params': params}, xs))


def predict_proba_softmax_bma(apply, params_sample, xs):
    """Predict probabilities for softmax with BMA prediction."""
    return vmap(
        lambda params: predict_proba_softmax_map(apply, params, xs)
    )(params_sample).mean(axis=0)


def predict_sigmoid_map(apply, params, xs):
    """Predict for sigmoid with MAP prediction."""
    return predict_proba_sigmoid_map(apply, params, xs) >= 0.5


def predict_sigmoid_bma(apply, params_sample, xs):
    """Predict for sigmoid with BMA prediction."""
    return predict_proba_sigmoid_bma(apply, params_sample, xs) >= 0.5


def predict_softmax_map(apply, params, xs):
    """Predict for softmax with MAP prediction."""
    return predict_proba_softmax_map(apply, params, xs).argmax(axis=-1)


def predict_softmax_bma(apply, params_sample, xs):
    """Predict for softmax with BMA prediction."""
    return predict_proba_softmax_bma(apply, params_sample, xs).argmax(axis=-1)


def accuracy_sigmoid_map(apply, params, batch_size, xs, ys):
    """Evaluate accuracy for sigmoid with MAP prediction."""
    correct = 0.0
    for xs_batch, ys_batch in iter_batches(1, batch_size, xs, ys):
        correct += (predict_sigmoid_map(
            apply, params, xs_batch) == ys_batch
        ).sum()
    return correct.item() / len(ys)


def accuracy_sigmoid_bma(apply, params_sample, batch_size, xs, ys):
    """Evaluate accuracy for softmax with BMA prediction."""
    correct = 0.0
    for xs_batch, ys_batch in iter_batches(1, batch_size, xs, ys):
        correct += (
            predict_sigmoid_bma(apply, params_sample, xs_batch) == ys_batch
        ).sum()
    return correct.item() / len(ys)


def accuracy_softmax_map(apply, params, batch_size, xs, ys):
    """Evaluate accuracy for softmax with MAP prediction."""
    correct = 0.0
    for xs_batch, ys_batch in iter_batches(1, batch_size, xs, ys):
        correct += (
            predict_softmax_map(apply, params, xs_batch) == ys_batch
        ).sum()
    return correct.item() / len(ys)


def accuracy_softmax_bma(apply, params_sample, batch_size, xs, ys):
    """Evaluate accuracy for softmax with BMA prediction."""
    correct = 0.0
    for xs_batch, ys_batch in iter_batches(1, batch_size, xs, ys):
        correct += (
            predict_softmax_bma(apply, params_sample, xs_batch) == ys_batch
        ).sum()
    return correct.item() / len(ys)
