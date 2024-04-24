"""Evaluation for sigmoid-activated models."""

from jax.nn import sigmoid

from torchds import fetch


def predict(state, x):
    return sigmoid(state.apply_fn({'params': state.params}, x))[:, 0] >= 0.5


def accuracy(state, batches):
    correct = 0
    count = 0
    for x, y in batches:
        correct += (predict(state, x) == y).sum()
        count += len(y)
    return correct / count
