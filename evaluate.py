"""Evaluation module."""

from jax.nn import sigmoid, softmax

from dataio import iter_batches


def predict(multiclass, state, x):
    """Predict array `x` using state."""
    out = state.apply_fn({'params': state.params}, x)
    if multiclass:
        return softmax(out).argmax(axis=-1)
    return sigmoid(out[:, 0]) >= 0.5


def accuracy(multiclass, state, batch_size, x, y):
    """Evaluate accuracy on arrays `x` and `y`."""
    correct = 0.0
    for x_batch, y_batch in iter_batches(1, batch_size, x, y):
        correct += (predict(multiclass, state, x_batch) == y_batch).sum()
    return correct.item() / len(y)
