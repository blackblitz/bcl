"""Training package."""

from .base import Finetuning
from .replay import Joint, GDumb
# from .svi import (
#     GaussianVCL, GaussianMixtureVCL,
#     BalancedRandomCoresetGaussianSFSVI,
#     BalancedRandomCoresetGaussianMixtureSFSVI
# )

__all__ = [
    'Finetuning',
    'Joint',
    'GDumb',
    # 'GaussianVCL',
    # 'GaussianMixtureVCL',
    # 'BalancedRandomCoresetGaussianSFSVI',
    # 'BalancedRandomCoresetGaussianMixtureSFSVI'
]
