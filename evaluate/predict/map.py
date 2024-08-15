"""Maximum-A-Posteriori predictors."""

from dataclasses import dataclass

from . import Predictor, SigmoidMixin, SoftmaxMixin


@dataclass
class MAPPredictor(Predictor):
    """MAP Predictor."""

    params: dict

    def predict_proba(self, xs):
        """Predict class probabilities."""
        return self._activate(self._apply(self.apply, self.params, xs))

    def predict(self, xs):
        """Predict a class."""
        return self._decide(self.predict_proba(xs))


@dataclass
class SigmoidMAPPredictor(SigmoidMixin, MAPPredictor):
    """Sigmoid MAP Predictor."""


@dataclass
class SoftmaxMAPPredictor(SoftmaxMixin, MAPPredictor):
    """Softmax MAP Predictor."""
