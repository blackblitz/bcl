"""Stateful predictors."""

from jax import random
import jax.numpy as jnp
import orbax.checkpoint as ocp

from .predictor import MAPMixin


class MAPPredictor(MAPMixin):
    """Predictor."""

    def __init__(self, model, model_spec, immutables, params, var):
        """Initialize self."""
        self.model = model
        self.model_spec = model_spec
        self.immutables = immutables
        self.params = params
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
        return cls(model, model_spec, immutables, var.pop('params'), var)

    def apply(self, xs):
        """Apply on the data."""
        return self.model.apply(
            {'params': self.params} | self.var, xs, train=False
        )
