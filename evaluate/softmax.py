"""Evaluation for softmax-activated models."""

from jax.nn import softmax

from torchds import fetch


def predict(state, x):
    return softmax(state.apply_fn({'params': state.params}, x)).argmax(axis=-1)


def accuracy(batch_size, state, dataset):
    correct = 0
    for x, y in fetch(dataset, 1, batch_size):
        correct += (predict(state, x) == y).sum()
    return correct / len(dataset)
