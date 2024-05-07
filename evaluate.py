"""Evalution module."""

from jax.nn import sigmoid, softmax

from dataio import iter_batches


def predict_sigmoid(state, x):
    return sigmoid(state.apply_fn({'params': state.params}, x))[:, 0] >= 0.5


def predict_softmax(state, x):
    return softmax(state.apply_fn({'params': state.params}, x)).argmax(axis=-1)


def accuracy(predict, batch_size, state, x, y):
    correct = 0
    for x_batch, y_batch in iter_batches(1, batch_size, x, y):
        correct += (predict(state, x_batch) == y_batch).sum()
    return correct / len(y)
