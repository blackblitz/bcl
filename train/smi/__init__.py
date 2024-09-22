"""Sequential MAP inference."""

from .replay import GDumb, TICReplay
from .simple import (
    AutodiffQuadraticConsolidation, ElasticWeightConsolidation,
    Finetuning, Joint, NeuralConsolidation, SynapticIntelligence
)

__all__ = [
    'AutodiffQuadraticConsolidation',
    'ElasticWeightConsolidation',
    'Finetuning',
    'GDumb',
    'Joint',
    'NeuralConsolidation',
    'SynapticIntelligence',
    'TICReplay'
]
