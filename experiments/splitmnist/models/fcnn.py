"""Fully-connected neural network for Split MNIST."""

from flax import linen as nn


class Main(nn.Module):
    """Main model."""

    @nn.compact
    def __call__(self, x):
        """Apply model."""
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(20)(x)
        x = nn.swish(x)
        x = nn.Dense(20)(x)
        x = nn.swish(x)
        return nn.Dense(10)(x)


class Consolidator(nn.Module):
    """Consolidator model."""

    @nn.compact
    def __call__(self, x):
        """Apply model."""
        x = nn.Dense(200)(x)
        x = nn.swish(x)
        x = nn.Dense(200)(x)
        x = nn.swish(x)
        x = nn.Dense(1)(x)
        return x
