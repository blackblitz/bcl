"""ISIC datasets."""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from .datasets import ArrayDataset, csplit, setdsattr


def preprocess(src):
    """Preprocess an ISIC dataset."""
    metadata = pd.read_csv(f'{src}/metadata.csv')
    trainmd, testmd = train_test_split(
        metadata, test_size=0.2,
        random_state=1337, stratify=metadata['diagnosis']
    )
    trainmd, valmd = train_test_split(
        trainmd, test_size=0.2,
        random_state=1337, stratify=trainmd['diagnosis']
    )
    return trainmd, valmd, testmd


def resize(size):
    """Read image from path, resize and convert to numpy array."""
    def transform(path):
        return np.asarray(
            Image.open(path).resize(size, resample=Image.Resampling.LANCZOS)
        ) / 255.0

    return transform


def makeds(src, classes, metadata):
    """Make an ISIC dataset from metadata."""
    class_to_index = {c: i for i, c in enumerate(classes)}
    metadata = metadata.loc[lambda df: df['diagnosis'].isin(classes)]
    paths = metadata['isic_id'].apply(lambda x: src / f'{x}.jpg').values
    cats = metadata['diagnosis'].apply(class_to_index.get).values
    return ArrayDataset(paths, cats, fx=resize((32, 32)))


def ts(src, classes, classess):
    """Generate a task sequence from a PyTorch Dataset constructor."""
    trainmd, valmd, testmd = preprocess(src)
    trainds = makeds(src, classes, trainmd)
    valds = makeds(src, classes, valmd)
    testds = makeds(src, classes, testmd)
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
                    [y for _, y in ds],
                    minlength=len(classes)
                ).tolist()
                for ds in dss
            ] for split, dss in tsdata.items()
        },
        'input_shape': testds[0][0].shape,
        'length': len(classess)
    }
    return tsdata, tsmetadata


def bcn12():
    """Make BCN12."""
    src = Path('data/BCN20000')
    classes = [
        'actinic keratosis', 'basal cell carcinoma', 'dermatofibroma',
        'melanoma', 'melanoma metastasis', 'nevus', 'other', 'scar',
        'seborrheic keratosis', 'solar lentigo', 'squamous cell carcinoma',
        'vascular lesion'
    ]
    classess = [list(range(12))]
    return ts(src, classes, classess)
    trainmd, valmd, testmd = preprocess(src)
    trainds = makeds(src, classes, trainmd)
    valds = makeds(src, classes, valmd)
    testds = makeds(src, classes, testmd)
    return (
        {
            'training': [trainds],
            'validation': [valds],
            'testing': [testds]
        },
        {
            'classes': classes,
            'input_shape': testds[0][0].shape,
            'length': 1
        }
    )


def cisplitham8():
    """Make class-incremental Split HAM8."""
    src = Path('data/HAM10000')
    classes = [
        'actinic keratosis', 'basal cell carcinoma', 'dermatofibroma',
        'melanoma', 'nevus', 'pigmented benign keratosis',
        'squamous cell carcinoma', 'vascular lesion'
    ]
    classess = [[0, 1], [2, 3], [4, 5], [6, 7]]
    return ts(src, classes, classess)


def displitham6():
    """Make domain-incremental Split HAM6."""
    src = Path('data/HAM10000')
    classes = [
        'pigmented benign keratosis', 'squamous cell carcinoma',
        'dermatofibroma', 'basal cell carcinoma',
        'nevus', 'melanoma'
    ]
    classess = [[0, 1], [2, 3], [4, 5]]
    tsdata, tsmetadata = ts(src, classes, classess)
    for split in tsdata:
        setdsattr(tsdata[split][0], 'fy', lambda x: x % 2)
    tsmetadata['classes'] = ['benign', 'malignant']
    tsmetadata['counts'] = {
        split: [
            np.bincount(
                [y for _, y in ds], minlength=len(tsmetadata['classes'])
            ).tolist()
            for ds in dss
        ] for split, dss in tsdata.items()
    }
    return tsdata, tsmetadata
