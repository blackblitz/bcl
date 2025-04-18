"""Models."""

from dataclasses import dataclass
from enum import Enum, unique

cnn = ['CNN4']
fecnn = ['FECNN4', 'FECNN6', 'FEResNet18']
fcnn = ['FCNN1', 'FCNN2', 'FCNN3']

module_map = (
    dict.fromkeys(fecnn, 'bcl.models.fecnn')
    | dict.fromkeys(cnn, 'bcl.models.cnn')
    | dict.fromkeys(fcnn, 'bcl.models.fcnn')
)


@unique
class NLL(Enum):
    """Negative log likelihood."""

    HUBER = 1
    L2 = 2
    SIGMOID_CROSS_ENTROPY = 3
    SOFTMAX_CROSS_ENTROPY = 4


@dataclass
class ModelSpec:
    """Model specification."""

    nll: NLL
    in_shape: list[int]
    out_shape: list[int]
    cratio: list[float]
    cscale: float
