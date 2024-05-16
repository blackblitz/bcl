"""Scikit-learn dataset."""

from jax import random
from torch.utils.data import Dataset


class SklearnDataset(Dataset):
    """Scikit-learn dataset."""

    def __init__(
        self, load, train=True, transform=None, target_transform=None
    ):
        """Initialize self."""
        self.transform = transform
        self.target_transform = target_transform
        data = load()
        mask = random.bernoulli(
            random.PRNGKey(1337), p=0.2, shape=data['target'].shape
        )
        if train:
            mask = ~mask
        self.x = data['data'][mask]
        self.y = data['target'][mask]

    def __len__(self):
        """Return the number of rows."""
        return len(self.y)

    def __getitem__(self, idx):
        """Get row by index."""
        x, y = self.x[idx], self.y[idx]
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)
        return x, y
