"""Predicting functions."""

from jax import random, tree_util, vmap
from jax.nn import sigmoid, softmax
import jax.numpy as jnp
import orbax.checkpoint as ocp

from .base import NNType
from .probability import (
    gauss_param, gsgauss_param, gauss_sample, gsgauss_sample
)
from .state.functions import init, gauss_init, gsgauss_init


class MAPPredictor:
    """Maximum-a-posteriori predictor."""

    def __init__(self, model, params, immutables, metadata):
        """Initialize self."""
        self.model = model
        self.params = params
        self.immutables = immutables
        self.metadata = metadata

    @classmethod
    def from_checkpoint(cls, model, path, immutables, metadata):
        """Initialize from a checkpoint."""
        with ocp.StandardCheckpointer() as ckpter:
            params = ckpter.restore(
                path,
                target=init(
                    random.PRNGKey(1337), model, metadata['input_shape']
                )
            )
        return cls(model, params, immutables, metadata)

    def __call__(self, xs, decide=True):
        """Predict the data."""
        nntype = NNType[self.immutables['nntype']]
        match nntype:
            case NNType.SIGMOID:
                proba = sigmoid(
                    self.model.apply({'params': self.params}, xs)[:, 0]
                )
                return proba >= 0.5 if decide else proba
            case NNType.SOFTMAX:
                proba = softmax(self.model.apply({'params': self.params}, xs))
                return proba.argmax(axis=-1) if decide else proba


class BMAPredictor:
    """Bayesian-model-averaging predictor."""

    def __init__(self, model, param_sample, immutables, metadata):
        """Initialize self."""
        self.model = model
        self.param_sample = param_sample
        self.immutables = immutables
        self.metadata = metadata

    @classmethod
    def from_checkpoint(cls, model, path, immutables, metadata):
        """Initialize from a checkpoint."""
        with ocp.StandardCheckpointer() as ckpter:
            param_sample = ckpter.restore(
                path,
                target=tree_util.tree_map(
                    lambda x: jnp.repeat(
                        jnp.expand_dims(x, 0), immutables['sample_size'],
                        axis=0
                    ),
                    init(
                        random.PRNGKey(1337),
                        model,
                        metadata['input_shape']
                    )
                )
            )
        return cls(model, param_sample, immutables, metadata)

    def __call__(self, xs, decide=True):
        """Predict the data."""
        nntype = NNType[self.immutables['nntype']]
        match nntype:
            case NNType.SIGMOID:
                proba = vmap(
                    lambda params: sigmoid(
                        self.apply({'params': params}, xs)[:, 0]
                    )
                )(self.param_sample).mean(axis=0)
                return proba >= 0.5 if decide else proba
            case NNType.SOFTMAX:
                proba = vmap(
                    lambda params: softmax(self.apply({'params': params}, xs))
                )(self.param_sample).mean(axis=0)
                return proba.argmax(axis=-1) if decide else proba


class GaussPredictor(BMAPredictor):
    """Gaussian-variational-inference predictor."""

    def __init__(self, model, params, immutables, metadata):
        """Initialize self."""
        self.model = model
        key = random.PRNGKey(1337)
        self.param_sample = gauss_param(
            params,
            gauss_sample(
                key,
                immutables['sample_size'],
                init(key, model, metadata['input_shape'])
            )
        )
        self.immutables = immutables
        self.metadata = metadata

    @classmethod
    def from_checkpoint(cls, model, path, immutables, metadata):
        """Initialize from a checkpoint."""
        with ocp.StandardCheckpointer() as ckpter:
            params = ckpter.restore(
                path, target=gauss_init(
                    random.PRNGKey(1337), model, metadata['input_shape']
                )
            )
        return cls(model, params, immutables, metadata)


class GSGaussPredictor(BMAPredictor):
    """Gumbel-softmax-Gaussian-mixture-variational-inference predictor."""

    def __init__(self, model, params, immutables, metadata):
        """Initialize self."""
        self.model = model
        key = random.PRNGKey(1337)
        self.param_sample = gsgauss_param(
            params,
            gsgauss_sample(
                key,
                immutables['sample_size'],
                immutables['n_comp'],
                init(key, model, metadata['input_shape'])
            )
        )
        self.immutables = immutables
        self.metadata = metadata

    @classmethod
    def from_checkpoint(cls, model, path, immutables, metadata):
        """Initialize from a checkpoint."""
        with ocp.StandardCheckpointer() as ckpter:
            params = ckpter.restore(
                path, target=gsgauss_init(
                    random.PRNGKey(1337),
                    model,
                    immutables['n_comp'],
                    metadata['input_shape']
                )
            )
        return cls(model, params, immutables, metadata)
