"""Sequential variational inference."""

from .replay import GMSFSVI, GSFSVI
from .simple import GMVCL, GVCL

__all__ = [
    'GMSFSVI',
    'GMVCL',
    'GSFSVI',
    'GVCL'
]
