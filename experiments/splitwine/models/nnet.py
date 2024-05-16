"""Neural network for Split Wine."""

from flax import linen as nn


class Main(nn.Module):
    """Main model."""

    @nn.compact
    def __call__(self, x):
        """Apply model."""
        x = nn.Dense(8)(x)
        x = nn.swish(x)
        x = nn.Dense(3)(x)
        return x


class Consolidator(nn.Module):
    """Consolidator model."""

    @nn.compact
    def __call__(self, x):
        """Apply model."""
        x = nn.Dense(2000)(x)
        x = nn.swish(x)
        x = nn.Dense(2000)(x)
        x = nn.swish(x)
        x = nn.Dense(1)(x)
        return x
