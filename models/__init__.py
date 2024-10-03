"""Models."""

from dataclasses import dataclass
from enum import Enum, unique

cnn = ['CNN4']
fecnn = ['FECNN4', 'FECNN7']
fcnn = ['FCNN1', 'FCNN2', 'FCNN3']

module_map = (
    dict.fromkeys(fecnn, 'models.stateful.fecnn')
    | dict.fromkeys(cnn, 'models.stateless.cnn')
    | dict.fromkeys(fcnn, 'models.stateless.fcnn')
)


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
