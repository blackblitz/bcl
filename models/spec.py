"""Model specification."""

from dataclasses import dataclass
from enum import Enum, unique


@unique
class NLL(Enum):
    """Negative log likelihood."""

    SIGMOID_CROSS_ENTROPY = 1
    SOFTMAX_CROSS_ENTROPY = 2


@dataclass
class ModelSpec:
    """Model specification."""

    nll: NLL
    in_shape: list[int]
    out_shape: list[int]
