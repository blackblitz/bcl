"""CNN with tanh-output features for pre-training for Split MNIST."""

from flax import linen as nn


class Model(nn.Module):
    def setup(self):
        self.feature_extractor = FeatureExtractor()
        self.classifier = nn.Dense(26)

    def __call__(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x


class FeatureExtractor(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(32, (3, 3))(x)
        x = nn.swish(x)
        x = nn.avg_pool(x, (2, 2), strides=(2, 2))
        x = nn.Conv(32, (3, 3))(x)
        x = nn.swish(x)
        x = nn.avg_pool(x, (2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(32)(x)
        x = nn.tanh(x)
        return x
