"""Evaluation for sigmoid-activated models."""

from jax.nn import sigmoid

from datasets import fetch


def predict(state, x):
    return sigmoid(state.apply_fn({'params': state.params}, x))[:, 0] >= 0.5


def accuracy(batch_size, state, dataset):
    correct = 0
    for x, y in fetch(dataset, 1, batch_size):
        correct += (predict(state, x) == y).sum()
    return correct / len(dataset)
