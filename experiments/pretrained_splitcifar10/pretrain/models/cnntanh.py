"""CNN with tanh-output features for pre-training for Split CIFAR-10."""

from flax import linen as nn


class Model(nn.Module):
    """Model."""

    def setup(self):
        """Set up feature extractor and classifier."""
        self.feature_extractor = FeatureExtractor()
        self.classifier = nn.Dense(100)

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
        x = nn.Conv(128, (3, 3))(x)
        x = nn.swish(x)
        x = nn.Conv(128, (3, 3))(x)
        x = nn.swish(x)
        x = nn.max_pool(x, (2, 2), strides=(2, 2))
        x = nn.Conv(256, (3, 3))(x)
        x = nn.swish(x)
        x = nn.Conv(256, (3, 3))(x)
        x = nn.swish(x)
        x = nn.max_pool(x, (2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(256)(x)
        x = nn.swish(x)
        x = nn.Dense(128)(x)
        x = nn.tanh(x)
        return x
