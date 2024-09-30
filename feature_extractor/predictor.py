"""Predictor."""

from jax import nn, random
import jax.numpy as jnp
import orbax.checkpoint as ocp

from models import NLL


class Predictor:
    """Predictor."""

    def __init__(self, model, model_spec, immutables, var):
        """Initialize self."""
        self.model = model
        self.model_spec = model_spec
        self.immutables = immutables
        self.var = var

    @classmethod
    def from_checkpoint(cls, model, model_spec, immutables, path):
        """Initialize from a checkpoint."""
        with ocp.StandardCheckpointer() as ckpter:
            var = ckpter.restore(
                path,
                target=model.init(
                    random.PRNGKey(1337),
                    jnp.zeros((1, *model_spec.in_shape)),
                    train=False
                )
            )
        return cls(model, model_spec, immutables, var)

    def __call__(self, xs, decide=True):
        """Predict the data."""
        match self.model_spec.nll:
            case NLL.SIGMOID_CROSS_ENTROPY:
                proba = nn.sigmoid(
                    self.model.apply(self.var, xs, train=False)[:, 0]
                )
                return proba >= 0.5 if decide else proba
            case NLL.SOFTMAX_CROSS_ENTROPY:
                proba = nn.softmax(
                    self.model.apply(self.var, xs, train=False)
                )
                return proba.argmax(axis=-1) if decide else proba
