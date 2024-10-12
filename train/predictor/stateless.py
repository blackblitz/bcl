"""Stateless predictor."""

from jax import random, vmap
import jax.numpy as jnp
import orbax.checkpoint as ocp

from ..probability import (
    gauss_param, gsgauss_param,
    gauss_sample, gsgauss_sample
)
from ..training.init import init, gauss_init, gsgauss_init

from .predictor import MAPMixin, BMAMixin


class MAPPredictor(MAPMixin):
    """Maximum-a-posteriori predictor."""

    def __init__(self, model, model_spec, immutables, params):
        """Initialize self."""
        self.model = model
        self.model_spec = model_spec
        self.immutables = immutables
        self.params = params

    @classmethod
    def from_checkpoint(cls, model, model_spec, immutables, path):
        """Initialize from a checkpoint."""
        with ocp.StandardCheckpointer() as ckpter:
            params = ckpter.restore(
                path,
                target=init(
                    random.PRNGKey(1337), model, model_spec.in_shape
                )
            )
        return cls(model, model_spec, immutables, params)

    def apply(self, xs):
        """Apply on the data."""
        return self.model.apply({'params': self.params}, xs)


class BMAPredictor(BMAMixin):
    """Bayesian-model-averaging predictor."""

    def __init__(self, model, model_spec, immutables, param_sample):
        """Initialize self."""
        self.model = model
        self.model_spec = model_spec
        self.immutables = immutables
        self.param_sample = param_sample

    @classmethod
    def from_checkpoint(cls, model, model_spec, immutables, path):
        """Initialize from a checkpoint."""
        with ocp.StandardCheckpointer() as ckpter:
            target = init(
                random.PRNGKey(1337), model, model_spec.in_shape
            )
            param_sample = ckpter.restore(
                path,
                target=vmap(
                    lambda x: target
                )(jnp.arange(immutables['sample_size']))
            )
        return cls(model, model_spec, immutables, param_sample)

    def apply(self, params, xs):
        """Apply on the data."""
        return self.model.apply({'params': params}, xs)


class GaussPredictor(BMAPredictor):
    """Gaussian-variational-inference predictor."""

    def __init__(self, model, model_spec, immutables, params):
        """Initialize self."""
        param_sample = gauss_param(
            params,
            gauss_sample(
                random.PRNGKey(immutables['seed']),
                immutables['sample_size'],
                init(random.PRNGKey(1337), model, model_spec.in_shape)
            )
        )
        super().__init__(model, model_spec, immutables, param_sample)

    @classmethod
    def from_checkpoint(cls, model, model_spec, immutables, path):
        """Initialize from a checkpoint."""
        with ocp.StandardCheckpointer() as ckpter:
            params = ckpter.restore(
                path, target=gauss_init(
                    random.PRNGKey(1337), model, model_spec.in_shape
                )
            )
        return cls(model, model_spec, immutables, params)


class GSGaussPredictor(BMAPredictor):
    """Gumbel-softmax-Gaussian-mixture-variational-inference predictor."""

    def __init__(self, model, model_spec, immutables, params):
        """Initialize self."""
        param_sample = gsgauss_param(
            params,
            gsgauss_sample(
                random.PRNGKey(immutables['seed']),
                immutables['sample_size'],
                immutables['n_comp'],
                init(random.PRNGKey(1337), model, model_spec.in_shape)
            )
        )
        super().__init__(model, model_spec, immutables, param_sample)

    @classmethod
    def from_checkpoint(cls, model, model_spec, immutables, path):
        """Initialize from a checkpoint."""
        with ocp.StandardCheckpointer() as ckpter:
            params = ckpter.restore(
                path, target=gsgauss_init(
                    random.PRNGKey(1337),
                    model,
                    immutables['n_comp'],
                    model_spec.in_shape,
                )
            )
        return cls(model, model_spec, immutables, params)
