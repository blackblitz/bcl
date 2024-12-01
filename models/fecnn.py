"""Feature-extracting convolutional neural networks."""

from flax import linen as nn
import jax.numpy as jnp


class FECNN4(nn.Module):
    """Feature-extracting convolutional neural network with 4 layers."""

    conv0: int
    conv1: int
    dense0: int
    dense1: int

    def setup(self):
        """Set up module."""
        self.tail = FE3(
            conv0=self.conv0,
            conv1=self.conv1,
            dense=self.dense0,
        )
        self.head = nn.Dense(self.dense1)

    def __call__(self, xs):
        """Apply module."""
        xs = self.tail(xs)
        xs = self.head(xs)
        return xs


class FECNN7(nn.Module):
    """Feature-extracting convolutional neural network with 7 layers."""

    conv0: int
    conv1: int
    conv2: int
    conv3: int
    dense0: int
    dense1: int
    dense2: int

    def setup(self):
        """Set up module."""
        self.tail = FE6(
            conv0=self.conv0, conv1=self.conv1,
            conv2=self.conv2, conv3=self.conv3,
            dense0=self.dense0, dense1=self.dense1,
        )
        self.head = nn.Dense(self.dense2)

    def __call__(self, xs):
        """Apply module."""
        xs = self.tail(xs)
        xs = self.head(xs)
        return xs


class FEResNet18(nn.Module):
    """Feature-extracting ResNet-18."""

    conv0: int
    conv1: int
    conv2: int
    conv3: int
    conv4: int
    dense: int

    def setup(self):
        """Set up module."""
        self.tail = FERes17(
            conv0=self.conv0, conv1=self.conv1,
            conv2=self.conv2, conv3=self.conv3,
            conv4=self.conv4
        )
        self.head = nn.Dense(self.dense)

    def __call__(self, xs):
        """Apply module."""
        xs = self.tail(xs)
        xs = self.head(xs)
        return xs


class FE3(nn.Module):
    """A feature extractor with 2 convolution layers and 1 dense layer."""

    conv0: int
    conv1: int
    dense: int

    @nn.compact
    def __call__(self, xs):
        """Apply module."""
        xs = nn.Conv(self.conv0, (3, 3))(xs)
        xs = nn.GroupNorm()(xs)
        xs = nn.swish(xs)
        xs = nn.avg_pool(xs, window_shape=(2, 2), strides=(2, 2))
        xs = nn.Conv(self.conv1, (3, 3))(xs)
        xs = nn.GroupNorm()(xs)
        xs = nn.swish(xs)
        xs = nn.avg_pool(xs, window_shape=(2, 2), strides=(2, 2))
        xs = jnp.reshape(xs, shape=(xs.shape[0], -1))
        xs = nn.Dense(self.dense)(xs)
        xs = nn.swish(xs)
        return xs


class FE6(nn.Module):
    """A feature extractor with 4 convolution layers and 2 dense layers."""

    conv0: int
    conv1: int
    conv2: int
    conv3: int
    dense0: int
    dense1: int

    @nn.compact
    def __call__(self, xs):
        """Apply module."""
        xs = nn.Conv(self.conv0, (3, 3))(xs)
        xs = nn.GroupNorm()(xs)
        xs = nn.swish(xs)
        xs = nn.Conv(self.conv1, (3, 3))(xs)
        xs = nn.GroupNorm()(xs)
        xs = nn.swish(xs)
        xs = nn.avg_pool(xs, window_shape=(2, 2), strides=(2, 2))
        xs = nn.Conv(self.conv2, (3, 3))(xs)
        xs = nn.GroupNorm()(xs)
        xs = nn.swish(xs)
        xs = nn.Conv(self.conv3, (3, 3))(xs)
        xs = nn.GroupNorm()(xs)
        xs = nn.swish(xs)
        xs = nn.avg_pool(xs, window_shape=(2, 2), strides=(2, 2))
        xs = jnp.reshape(xs, shape=(xs.shape[0], -1))
        xs = nn.Dense(self.dense0)(xs)
        xs = nn.swish(xs)
        xs = nn.Dense(self.dense1)(xs)
        xs = nn.swish(xs)
        return xs


class FERes17(nn.Module):
    """A feature extractor with 17 convolutional layers for ResNet."""
    
    conv0: int
    conv1: int
    conv2: int
    conv3: int
    conv4: int

    @nn.compact
    def __call__(self, xs):
        """Apply module."""
        xs = nn.Conv(self.conv0, (3, 3))(xs)
        xs = nn.GroupNorm()(xs)
        xs = nn.swish(xs)
        xs = ResidualBlock(self.conv1)(xs)
        xs = ResidualBlock(self.conv1)(xs)
        xs = nn.avg_pool(xs, window_shape=(2, 2), strides=(2, 2))
        xs = ResidualBlock(self.conv2)(xs)
        xs = ResidualBlock(self.conv2)(xs)
        xs = nn.avg_pool(xs, window_shape=(2, 2), strides=(2, 2))
        xs = ResidualBlock(self.conv3)(xs)
        xs = ResidualBlock(self.conv3)(xs)
        xs = nn.avg_pool(xs, window_shape=(2, 2), strides=(2, 2))
        xs = ResidualBlock(self.conv4)(xs)
        xs = ResidualBlock(self.conv4)(xs)
        xs = jnp.mean(xs, axis=(1, 2))
        return xs


class ResidualBlock(nn.Module):
    """A convolutional residual block used in ResNet."""

    conv: int

    @nn.compact
    def __call__(self, xs):
        """Apply module."""
        res = nn.Conv(self.conv, (3, 3))(xs)
        res = nn.GroupNorm()(res)
        res = nn.swish(res)
        res = nn.Conv(self.conv, (3, 3))(res)
        res = nn.GroupNorm()(res)
        if xs.shape != res.shape:
            xs = nn.Conv(self.conv, (1, 1))(xs)
            xs = nn.GroupNorm()(xs)
        xs = xs + res
        xs = nn.swish(xs)
        return xs


class DownsamplingResidualBlock(nn.Module):
    """A downsampling convolutional residual block used in ResNet."""

    conv: int

    @nn.compact
    def __call__(self, xs):
        """Apply module."""
        res = nn.Conv(self.conv, (3, 3))(xs)
        res = nn.GroupNorm()(res)
        res = nn.swish(res)
        res = nn.Conv(self.conv, (3, 3), strides=(2, 2))(res)
        res = nn.GroupNorm()(res)
        if xs.shape != res.shape:
            xs = nn.Conv(self.conv, (1, 1), strides=(2, 2))(xs)
            xs = nn.GroupNorm()(xs)
        xs = xs + res
        xs = nn.swish(xs)
        return xs
