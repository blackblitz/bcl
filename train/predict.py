"""Predicting functions."""

from jax import random, vmap
from jax.nn import sigmoid, softmax
import jax.numpy as jnp
import orbax.checkpoint as ocp

from models import NLL

from .probability import (
    bern_entr, cat_entr,
    gauss_param, gsgauss_param,
    gauss_sample, gsgauss_sample
)
from .state.functions import init, gauss_init, gsgauss_init


class MAPPredictor:
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

    def __call__(self, xs, decide=True):
        """Predict the data."""
        match self.model_spec.nll:
            case NLL.SIGMOID_CROSS_ENTROPY:
                proba = sigmoid(
                    self.model.apply({'params': self.params}, xs)[:, 0]
                )
                return proba >= 0.5 if decide else proba
            case NLL.SOFTMAX_CROSS_ENTROPY:
                proba = softmax(self.model.apply({'params': self.params}, xs))
                return proba.argmax(axis=-1) if decide else proba

    def entropy(self, xs):
        """Calculate the entropy of predictions."""
        match self.model_spec.nll:
            case NLL.SIGMOID_CROSS_ENTROPY:
                return bern_entr(self(xs, decide=False))
            case NLL.SOFTMAX_CROSS_ENTROPY:
                return cat_entr(self(xs, decide=False))


class BMAPredictor:
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

    def sample(self, xs):
        """Return prediction samples."""
        match self.model_spec.nll:
            case NLL.SIGMOID_CROSS_ENTROPY:
                return vmap(
                    lambda params: sigmoid(
                        self.model.apply({'params': params}, xs)[:, 0]
                    )
                )(self.param_sample)
            case NLL.SOFTMAX_CROSS_ENTROPY:
                return vmap(
                    lambda params: softmax(
                        self.model.apply({'params': params}, xs))
                )(self.param_sample)

    def __call__(self, xs, decide=True):
        """Predict the data."""
        match self.model_spec.nll:
            case NLL.SIGMOID_CROSS_ENTROPY:
                proba = self.sample(xs).mean(axis=0)
                return proba >= 0.5 if decide else proba
            case NLL.SOFTMAX_CROSS_ENTROPY:
                proba = self.sample(xs).mean(axis=0)
                return proba.argmax(axis=-1) if decide else proba

    def entropy(self, xs):
        """Calculate the entropy of predictions."""
        match self.model_spec.nll:
            case NLL.SIGMOID_CROSS_ENTROPY:
                return bern_entr(self.sample(xs).mean(axis=0))
            case NLL.SOFTMAX_CROSS_ENTROPY:
                return cat_entr(self.sample(xs).mean(axis=0))

    def mutual_information(self, xs):
        """Calculate the MI between the predictions and the parameters."""
        match self.model_spec.nll:
            case NLL.SIGMOID_CROSS_ENTROPY:
                return (
                    bern_entr(self.sample(xs).mean(axis=0))
                    - bern_entr(self.sample(xs)).mean(axis=0)
                )
            case NLL.SOFTMAX_CROSS_ENTROPY:
                return (
                    cat_entr(self.sample(xs).mean(axis=0))
                    - cat_entr(self.sample(xs)).mean(axis=0)
                )


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
