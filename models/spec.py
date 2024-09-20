"""Model specification."""

from dataclasses import dataclass
from enum import Enum, unique


@unique
class FinAct(Enum):
    """Final activation function."""
    SIGMOID = 1
    SOFTMAX = 2


@dataclass
class ModelSpec:
    """Model specification."""
    fin_act: FinAct
    in_shape: list[int]
    out_shape: list[int]
