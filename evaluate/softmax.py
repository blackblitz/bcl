"""Evaluation for softmax-activated models."""

from jax.nn import softmax

from torchds import fetch


def predict(state, x):
    return softmax(state.apply_fn({'params': state.params}, x)).argmax(axis=-1)


def accuracy(state, batches):
    correct = 0
    count = 0
    for x, y in batches:
        correct += (predict(state, x) == y).sum()
        count += len(y)
    return correct / count
