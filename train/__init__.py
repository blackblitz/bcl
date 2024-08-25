"""Training package."""

from .base import Finetuning
from .replay import Joint, RandomConcatReplay, RandomChoiceReplay
# from .svi import (
#     GaussianVCL, GaussianMixtureVCL,
#     BalancedRandomCoresetGaussianSFSVI,
#     BalancedRandomCoresetGaussianMixtureSFSVI
# )

__all__ = [
    'Finetuning',
    'Joint',
    'BalancedRandomCoresetReplay',
    'GaussianVCL',
    'GaussianMixtureVCL',
    'BalancedRandomCoresetGaussianSFSVI',
    'BalancedRandomCoresetGaussianMixtureSFSVI'
]
