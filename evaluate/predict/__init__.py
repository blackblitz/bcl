"""Predictors."""

from .bma import (
    SigmoidGVIBMAPredictor, SoftmaxGVIBMAPredictor,
    SigmoidGMVIBMAPredictor, SoftmaxGMVIBMAPredictor
)
from .map import SigmoidMAPPredictor, SoftmaxMAPPredictor

__all__ = [
    'SigmoidGVIBMAPredictor',
    'SoftmaxGVIBMAPredictor',
    'SigmoidGMVIBMAPredictor',
    'SoftmaxGMVIBMAPredictor',
    'SigmoidMAPPredictor',
    'SoftmaxMAPPredictor'
]
