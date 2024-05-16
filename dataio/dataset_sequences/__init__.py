"""Dataset sequences."""

from itertools import accumulate

from torch.utils.data import ConcatDataset


def accumulate_full(datasets):
    """Accumulate by concatenating the datasets."""
    return accumulate(datasets, func=lambda x, y: ConcatDataset([x, y]))
