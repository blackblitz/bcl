"""CNN with swish-output features for pre-training for Split MNIST."""

from flax import linen as nn


class Model(nn.Module):
    """Model."""

    def setup(self):
        """Set up feature extractor and classifier."""
        self.feature_extractor = FeatureExtractor()
        self.classifier = nn.Dense(26)

    def __call__(self, x):
        """Apply model."""
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x


class FeatureExtractor(nn.Module):
    """Feature extractor."""

    @nn.compact
    def __call__(self, x):
        """Apply model."""
        x = nn.Conv(32, (3, 3))(x)
        x = nn.swish(x)
        x = nn.avg_pool(x, (2, 2), strides=(2, 2))
        x = nn.Conv(32, (3, 3))(x)
        x = nn.swish(x)
        x = nn.avg_pool(x, (2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(32)(x)
        x = nn.swish(x)
        return x
