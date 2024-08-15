"""Softmax regression for Split Iris 2."""

from flax import linen as nn


class Model(nn.Module):
    """Model."""

    @nn.compact
    def __call__(self, x):
        """Apply model."""
        return nn.Dense(3)(x)
