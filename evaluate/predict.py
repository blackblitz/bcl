"""Predictors."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import partial
from typing import Callable


@dataclass
class Predictor(ABC):
    apply: Callable

    @abstractmethod
    def predict_proba(self, xs):

    @abstractmethod
    def decide(self, ps):

    @abstractmethod
    def predict(self, xs):


@dataclass
class MAPPredictor(Predictor):
    params: dict

    def predict(self, xs):
        return toolz.pipe(
            xs,
            partial(self.predict_proba, self.params),
            self.decide
        )


@dataclass
class VIBMAPredictor(Predictor):
    var_params: dict
    sample_key: int
    sample_size: int
    params_sample: field(init=False)

    def predict(self, xs):
        return toolz.pipe(
            xs,
            lambda xs: vmap(
                partial(self.predict_proba, xs=xs)
            )(self.params_sample).mean(axis=0),
            self.decide
        )


class SigmoidMixin:
    def predict_proba(self, params, xs):
        return sigmoid(self.apply({'params': params}, xs)[:, 0])

    def decide(self, ps):
        return ps >= 0.5


class SoftmaxMixin:
    def predict_proba(self, params, xs):
        return softmax(self.apply({'params': params}, xs))

    def decide(self, ps):
        return ps.argmax(axis=-1)


class GaussianMixin:
    def __post_init__(self):
        self.params_sample = GaussianSVIMixin.sample_params(
            self.var_params,
            GaussianSVIMixin.sample_std(
                self.sample_key, self.sample_size, self.var_params
            )
        )


class GaussianMixtureMixin:
    def __post_init__(self):
        self.params_sample = GaussianMixtureSVIMixin.sample_params(
            self.var_params,
            GaussianMixtureSVIMixin.sample_std(
                self.sample_key, self.sample_size, self.var_params
            )
        )


@dataclass
class SigmoidMAPPredictor(SigmoidMixin, MAPPredictor):


@dataclass
class SoftmaxMAPPredictor(SoftmaxMixin, MAPPredictor):


@dataclass
class SigmoidGVIBMAPredictor(SigmoidMixin, GaussianMixin, VIBMAPredictor):


@dataclass
class SoftmaxGVIBMAPredictor(SoftmaxMixin, GaussianMixin, VIBMAPredictor):


@dataclass
class SigmoidGMVIBMAPredictor(
    SigmoidMixin, GaussianMixtureMixin, VIBMAPredictor
):


@dataclass
class SoftmaxGMVIBMAPredictor(
    SoftmaxMixin, GaussianMixtureMixin, VIBMAPredictor
):
