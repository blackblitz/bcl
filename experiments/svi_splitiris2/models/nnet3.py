"""Neural network with 3 hidden nodes for Split Iris 2."""

from flax import linen as nn


class Model(nn.Module):
    """Model."""

    @nn.compact
    def __call__(self, x):
        """Apply model."""
        x = nn.Dense(3)(x)
        x = nn.swish(x)
        x = nn.Dense(3)(x)
        return x
