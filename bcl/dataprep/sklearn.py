"""Task sequences based on scikit-learn datasets."""

import numpy as np
from sklearn.datasets import load_iris, load_wine

from . import seed
from .datasets import ArrayDataset, csplit, rsplit


def ts(load, classess, fy=lambda x: x):
    """Generate a task sequence from scikit-learn load function."""
    dataset = load()
    dstrain = ArrayDataset(dataset['data'], dataset['target'])
    dstrain.fy = fy
    dstrain, dstest = rsplit(seed, 0.2, dstrain)
    dstrain, dsval = rsplit(seed, 0.2, dstrain)
    tsdata = {
        'training': csplit(classess, dstrain),
        'validation': csplit(classess, dsval),
        'testing': csplit(classess, dstest)
    }
    tsmetadata = {
        'classes': dataset.target_names.tolist(),
        'counts': {
            split: [
                np.bincount(
                    [y for _, y in ds],
                    minlength=len(dataset.target_names)
                ).tolist()
                for ds in dss
            ] for split, dss in tsdata.items()
        },
        'features': dataset.feature_names,
        'input_shape': [len(dataset.feature_names)],
        'input_min': np.min([x for x, _ in dstrain], axis=0).tolist(),
        'input_max': np.max([x for x, _ in dstrain], axis=0).tolist(),
        'length': len(classess)
    }
    return tsdata, tsmetadata


def cisplitiris():
    """Make Split Iris."""
    return ts(load_iris, np.arange(3))


def cisplit2diris():
    """Make Split 2D Iris."""
    def load():
        dataset = load_iris()
        dataset['data'] = dataset['data'][:, 2:4]
        dataset.feature_names = dataset.feature_names[2:4]
        return dataset

    return ts(load, np.arange(3))


def cisplitwine():
    """Make Split Wine."""
    return ts(load_wine, np.arange(3))
