"""Predicting functions."""

from jax import random, vmap
from jax.nn import sigmoid, softmax
import jax.numpy as jnp
import orbax.checkpoint as ocp

from models import FinAct

from .probability import (
    bern_entr, cat_entr,
    gauss_param, gsgauss_param,
    gauss_sample, gsgauss_sample
)
from .state.functions import init, gauss_init, gsgauss_init


class MAPPredictor:
    """Maximum-a-posteriori predictor."""

    def __init__(self, model, model_spec, params):
        """Initialize self."""
        self.model = model
        self.model_spec = model_spec
        self.params = params

    @classmethod
    def from_checkpoint(cls, model, model_spec, path):
        """Initialize from a checkpoint."""
        with ocp.StandardCheckpointer() as ckpter:
            params = ckpter.restore(
                path,
                target=init(
                    random.PRNGKey(1337), model, model_spec.in_shape
                )
            )
        return cls(model, model_spec, params)

    def __call__(self, xs, decide=True):
        """Predict the data."""
        match self.model_spec.fin_act:
            case FinAct.SIGMOID:
                proba = sigmoid(
                    self.model.apply({'params': self.params}, xs)[:, 0]
                )
                return proba >= 0.5 if decide else proba
            case FinAct.SOFTMAX:
                proba = softmax(self.model.apply({'params': self.params}, xs))
                return proba.argmax(axis=-1) if decide else proba

    def entropy(self, xs):
        """Calculate the entropy of predictions."""
        match self.model_spec.fin_act:
            case FinAct.SIGMOID:
                return bern_entr(self(xs, decide=False))
            case FinAct.SOFTMAX:
                return cat_entr(self(xs, decide=False))


class BMAPredictor:
    """Bayesian-model-averaging predictor."""

    def __init__(self, model, model_spec, param_sample):
        """Initialize self."""
        self.model = model
        self.model_spec = model_spec
        self.param_sample  = param_sample

    @classmethod
    def from_checkpoint(cls, model, model_spec, path):
        """Initialize from a checkpoint."""
        with ocp.StandardCheckpointer() as ckpter:
            target = init(
                random.PRNGKey(1337), model, model_spec.in_shape
            )
            param_sample = ckpter.restore(
                path,
                target=vmap(lambda x: target)(
                    jnp.arange(immutables['pred_sample_size'])
                )
            )
        return cls(model, model_spec, param_sample)

    def sample(self, xs):
        """Return prediction samples."""
        match self.model_spec.fin_act:
            case FinAct.SIGMOID:
                return vmap(
                    lambda params: sigmoid(
                        self.model.apply({'params': params}, xs)[:, 0]
                    )
                )(self.param_sample)
            case FinAct.SOFTMAX:
                return vmap(
                    lambda params: softmax(
                        self.model.apply({'params': params}, xs))
                )(self.param_sample)

    def __call__(self, xs, decide=True):
        """Predict the data."""
        match self.model_spec.fin_act:
            case FinAct.SIGMOID:
                proba = self.sample(xs).mean(axis=0)
                return proba >= 0.5 if decide else proba
            case FinAct.SOFTMAX:
                proba = self.sample(xs).mean(axis=0)
                return proba.argmax(axis=-1) if decide else proba

    def entropy(self, xs):
        """Calculate the entropy of predictions."""
        match self.model_spec.fin_act:
            case FinAct.SIGMOID:
                return bern_entr(self.sample(xs).mean(axis=0))
            case FinAct.SOFTMAX:
                return cat_entr(self.sample(xs).mean(axis=0))

    def mutual_information(self, xs):
        """Calculate the MI between the predictions and the parameters."""
        match self.model_spec.fin_act:
            case FinAct.SIGMOID:
                return (
                    bern_entr(self.sample(xs).mean(axis=0))
                    - bern_entr(self.sample(xs)).mean(axis=0)
                )
            case FinAct.SOFTMAX:
                return (
                    cat_entr(self.sample(xs).mean(axis=0))
                    - cat_entr(self.sample(xs)).mean(axis=0)
                )


class GaussPredictor(BMAPredictor):
    """Gaussian-variational-inference predictor."""

    def __init__(
        self, model, model_spec, params,
        key=random.PRNGKey(1337), sample_size=10
    ):
        """Initialize self."""
        param_sample = gauss_param(
            params,
            gauss_sample(
                key, sample_size,
                init(random.PRNGKey(1337), model, model_spec.in_shape)
            )
        )
        super().__init__(model, model_spec, param_sample)

    @classmethod
    def from_checkpoint(
        cls, model, model_spec, path,
        key=random.PRNGKey(1337), sample_size=10
    ):
        """Initialize from a checkpoint."""
        with ocp.StandardCheckpointer() as ckpter:
            params = ckpter.restore(
                path, target=gauss_init(
                    random.PRNGKey(1337), model, model_spec.in_shape
                )
            )
        return cls(
            model, model_spec, params, key=key, sample_size=sample_size
        )


class GSGaussPredictor(BMAPredictor):
    """Gumbel-softmax-Gaussian-mixture-variational-inference predictor."""

    def __init__(
        self, model, model_spec, n_comp, params,
        key=random.PRNGKey(1337), sample_size=10
    ):
        """Initialize self."""
        param_sample = gsgauss_param(
            params,
            gsgauss_sample(
                key, sample_size, n_comp,
                init(random.PRNGKey(1337), model, model_spec.in_shape)
            )
        )
        super().__init__(model, model_spec, param_sample)

    @classmethod
    def from_checkpoint(
        cls, model, model_spec, n_comp, path,
        key=random.PRNGKey(1337), sample_size=10
    ):
        """Initialize from a checkpoint."""
        with ocp.StandardCheckpointer() as ckpter:
            params = ckpter.restore(
                path, target=gsgauss_init(
                    random.PRNGKey(1337),
                    model,
                    n_comp,
                    model_spec.in_shape
                )
            )
        return cls(
            model, model_spec, n_comp, params,
            key=key, sample_size=sample_size
        )
