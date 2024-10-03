"""Predictor base classes and mixins."""

from jax import nn, vmap

from models import NLL

from ..probability import bern_entr, cat_entr


class MAPMixin:
    """Maximum-a-posteriori mixin."""

    def __call__(self, xs, decide=True):
        """Predict the data."""
        match self.model_spec.nll:
            case NLL.SIGMOID_CROSS_ENTROPY:
                proba = nn.sigmoid(self.apply(xs)[:, 0])
                return proba >= 0.5 if decide else proba
            case NLL.SOFTMAX_CROSS_ENTROPY:
                proba = nn.softmax(self.apply(xs))
                return proba.argmax(axis=-1) if decide else proba

    def entropy(self, xs):
        """Calculate the entropy of predictions."""
        match self.model_spec.nll:
            case NLL.SIGMOID_CROSS_ENTROPY:
                return bern_entr(self(xs, decide=False))
            case NLL.SOFTMAX_CROSS_ENTROPY:
                return cat_entr(self(xs, decide=False))


class BMAMixin:
    """Bayesian-model-averaging mixin."""

    def sample(self, xs):
        """Return prediction samples."""
        match self.model_spec.nll:
            case NLL.SIGMOID_CROSS_ENTROPY:
                return vmap(
                    lambda params: nn.sigmoid(self.apply(xs)[:, 0])
                )(self.param_sample)
            case NLL.SOFTMAX_CROSS_ENTROPY:
                return vmap(
                    lambda params: nn.softmax(self.apply(xs))
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
