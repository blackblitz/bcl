"""Evaluation module."""

from itertools import islice

import jax.numpy as jnp
from jax.nn import sigmoid, softmax
from torch.utils.data import DataLoader

from dataseqs import numpy_collate


def apply(state, x):
    return state.apply_fn({'params': state.params}, x)


def acc_sigmoid(state, dataset):
    correct = 0
    for x, y in DataLoader(dataset, batch_size=64, collate_fn=numpy_collate):
        correct += ((sigmoid(apply(state, x))[:, 0] >= 0.5) == y).sum()
    return correct / len(dataset)


def acc_softmax(state, dataset):
    correct = 0
    for x, y in DataLoader(dataset, batch_size=64, collate_fn=numpy_collate):
        correct += (softmax(apply(state, x)).argmax(axis=-1) == y).sum()
    return correct / len(dataset)


def aa_sigmoid(i, state, dataseq):
    return jnp.mean(jnp.array([
        acc_sigmoid(state, dataset) for dataset in islice(dataseq.test(), 0, i + 1)
    ]))


def aa_softmax(i, state, dataseq):
    return jnp.mean(jnp.array([
        acc_softmax(state, dataset) for dataset in islice(dataseq.test(), 0, i + 1)
    ]))