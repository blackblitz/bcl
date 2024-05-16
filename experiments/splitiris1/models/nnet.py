"""Neural network for Split Iris 1."""

from flax import linen as nn


class Main(nn.Module):
    """Main model."""

    @nn.compact
    def __call__(self, x):
        """Apply model."""
        x = nn.Dense(1, use_bias=False)(x)
        x = nn.swish(x)
        x = nn.Dense(1, use_bias=False)(x)
        return x


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
