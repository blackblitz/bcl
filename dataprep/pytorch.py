"""Task sequences based on PyTorch datasets."""

from pathlib import Path

import numpy as np
from torchvision.datasets import CIFAR10, CIFAR100, EMNIST, MNIST

from . import seed
from .datasets import csplit, rsplit, setdsattr

src = Path('data/pytorch')


def ts(constructor, classes, classess, **args):
    """Generate a task sequence from a PyTorch Dataset constructor."""
    trainds = constructor(src, download=True, train=True, **args)
    trainds, valds = rsplit(seed, 0.2, trainds)
    testds = constructor(src, download=True, train=False, **args)
    if classes is None:
        classes = testds.classes
    tsdata = {
        'training': csplit(classess, trainds),
        'validation': csplit(classess, valds),
        'testing': csplit(classess, testds)
    }
    tsmetadata = {
        'classes': classes,
        'counts': {
            split: [
                np.bincount(
                    [y for _, y in ds], minlength=len(classes)
                ).tolist()
                for ds in dss
            ] for split, dss in tsdata.items()
        },
        'input_shape': testds[0][0].shape,
        'length': len(classess)
    }
    return tsdata, tsmetadata


def emnistletters():
    """Make EMNIST Letters."""
    classes = [chr(c) for c in range(ord('a'), ord('z') + 1)]
    tsdata, tsmetadata = ts(
        EMNIST, classes, [list(range(26))],
        transform=lambda x: np.asarray(x)[:, :, None] / 255.0,
        target_transform=lambda x: x - 1,
        split='letters'
    )
    return tsdata, tsmetadata


def cifar100():
    """Make CIFAR-100."""
    return ts(
        CIFAR100, None, [list(range(100))],
        transform=lambda x: np.asarray(x) / 255.0
    )


def cisplitmnist():
    """Make class-incremental Split MNIST."""
    return ts(
        MNIST, None, [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]],
        transform=lambda x: np.asarray(x)[:, :, None] / 255.0
    )


def cisplitcifar10():
    """Make class-incremental Split CIFAR-10."""
    return ts(
        CIFAR10, None, [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]],
        transform=lambda x: np.asarray(x) / 255.0
    )


def displitmnist():
    """Make domain-incremental Split MNIST."""
    tsdata, tsmetadata = ts(
        MNIST, None, [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]],
        transform=lambda x: np.asarray(x)[:, :, None] / 255.0
    )
    for split in ['training', 'validation', 'testing']:
        setdsattr(tsdata[split][0], 'target_transform', lambda x: x % 2)
    tsmetadata['classes'] = ['even', 'odd']
    tsmetadata['counts'] = {
        split: [
            np.bincount(
                [y for _, y in ds], minlength=len(tsmetadata['classes'])
            ).tolist()
            for ds in dss
        ] for split, dss in tsdata.items()
    }
    return tsdata, tsmetadata


def displitcifar8():
    """Make domain-incremental Split CIFAR-8."""
    # (airplane, cat), (automobile, deer), (ship, dog), (truck, horse)
    classess = [[0, 3], [1, 4], [8, 5], [9, 7]]
    remap = {}
    for classes in classess:
        remap |= dict(zip(classes, range(len(classes))))
    tsdata, tsmetadata = ts(
        CIFAR10, None, classess,
        transform=lambda x: np.asarray(x) / 255.0
    )
    for split in ['training', 'validation', 'testing']:
        setdsattr(tsdata[split][0], 'target_transform', remap.get)
    tsmetadata['classes'] = ['vehicle', 'animal']
    tsmetadata['counts'] = {
        split: [
            np.bincount(
                [y for _, y in ds], minlength=len(tsmetadata['classes'])
            ).tolist()
            for ds in dss
        ] for split, dss in tsdata.items()
    }
    return tsdata, tsmetadata
