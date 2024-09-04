"""Metrics."""

from dataops.array import pass_batches


def accuracy(pass_size, predict, xs, ys):
    """Accuracy."""
    correct = 0
    for xs_batch, ys_batch in pass_batches(pass_size, xs, ys):
        correct += (predict(xs_batch) == ys_batch).sum()
    return correct.item() / len(ys)
