"""Metrics."""

from dataio import pass_batches
from dataio.datasets import to_arrays


def accuracy(batch_size, predictor, xs, ys):
    """Accuracy."""
    correct = 0
    for xs_batch, ys_batch in pass_batches(batch_size, xs, ys):
        correct += (predictor.predict(xs_batch) == ys_batch).sum()
    return correct.item() / len(ys)
