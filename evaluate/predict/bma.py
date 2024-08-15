"""Variational-inference Bayesian-model-averaging predictors."""

from dataclasses import dataclass, field

from jax import vmap

from . import Predictor, SigmoidMixin, SoftmaxMixin
from train.svi import GaussianSVIMixin, GaussianMixtureSVIMixin


@dataclass
class VIBMAPredictor(Predictor):
    """VIBMA predictor."""

    params: dict
    sample_key: int
    sample_size: int
    params_sample: dict = field(init=False)

    def predict_proba(self, xs):
        """Predict class probabilities."""
        return vmap(
            lambda params: self._activate(self._apply(self.apply, params, xs))
        )(self.params_sample).mean(axis=0)

    def predict(self, xs):
        """Predict a class."""
        return self._decide(self.predict_proba(xs))


class GaussianMixin:
    """Mixin for Gaussian variational inference."""

    def __post_init__(self):
        """Post-initialize self."""
        self.params_sample = GaussianSVIMixin.sample_params(
            self.params,
            GaussianSVIMixin.sample_std(
                self.sample_key, self.sample_size, self.params
            )
        )


class GaussianMixtureMixin:
    """Mixin for Gaussian-mixture variational inference."""

    def __post_init__(self):
        """Post-initialize self."""
        self.params_sample = GaussianMixtureSVIMixin.sample_params(
            self.params,
            GaussianMixtureSVIMixin.sample_std(
                self.sample_key, self.sample_size, self.params
            )
        )


@dataclass
class SigmoidGVIBMAPredictor(SigmoidMixin, GaussianMixin, VIBMAPredictor):
    """Sigmoid Gaussian VIBMA Predictor."""


@dataclass
class SoftmaxGVIBMAPredictor(SoftmaxMixin, GaussianMixin, VIBMAPredictor):
    """Softmax Gaussian VIBMA Predictor."""


@dataclass
class SigmoidGMVIBMAPredictor(
    SigmoidMixin, GaussianMixtureMixin, VIBMAPredictor
):
    """Sigmoid Gaussian-mixture VIBMA Predictor."""


@dataclass
class SoftmaxGMVIBMAPredictor(
    SoftmaxMixin, GaussianMixtureMixin, VIBMAPredictor
):
    """Softmax Gaussian-mixture VIBMA Predictor."""
