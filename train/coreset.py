"""Coreset."""

from abc import ABC, abstractmethod
from pathlib import Path

from jax import random, tree_util
import numpy as np
import zarr

from dataops.array import batch, get_pass_size, shuffle
from dataops.io import clear


class Coreset(ABC):
    """Abstract base class for a coreset."""

    def __init__(self, zarr_path, memmap_path, model_spec):
        """Initialize self."""
        self.zarr_path = Path(zarr_path)
        self.memmap_path = Path(memmap_path)
        self.model_spec = model_spec
        self.zarr = zarr.open(self.zarr_path, mode='w')
        self.memmap = None

    def empty(self):
        """Return empty arrays xs and ys."""
        return (
            np.empty(
                (0, *self.model_spec.in_shape), dtype=np.float32
            ),
            np.empty((0,), dtype=np.uint8)
        )

    def create_memmap(self):
        """Create memory-mapped files."""
        pass_size = get_pass_size(self.model_spec.in_shape)

        def write(name, obj):
            path = self.memmap_path / name
            if isinstance(obj, zarr.Group):
                path.mkdir()
            elif isinstance(obj, zarr.Array):
                array = np.lib.format.open_memmap(
                    path, mode='w+',
                    dtype=obj.dtype,
                    shape=obj.shape
                )
                for idx in batch(pass_size, np.arange(len(obj))):
                    array[idx] = obj[idx]

        self.memmap_path.mkdir(exist_ok=True)
        self.zarr.visititems(write)

        self.memmap = {}

        def read(name, obj):
            path = self.memmap_path / name
            if isinstance(obj, zarr.Array):
                self.memmap[name] = np.lib.format.open_memmap(path, mode='r')

        self.zarr.visititems(read)

    def delete_memmap(self):
        """Delete memory-mapped files."""
        clear(self.memmap_path)
        self.memmap_path.rmdir()

    @abstractmethod
    def update(self, key, xs, ys):
        """Update self."""


class JointCoreset(Coreset):
    """Joint coreset."""

    def __init__(self, zarr_path, memmap_path, model_spec):
        """Initialize self."""
        super().__init__(zarr_path, memmap_path, model_spec)
        self.zarr['xs'], self.zarr['ys'] = self.empty()

    def update(self, key, xs, ys):
        """Update self."""
        pass_size = get_pass_size(self.model_spec.in_shape)
        for indices in batch(pass_size, np.arange(len(ys))):
            self.zarr['xs'].append(xs[indices])
            self.zarr['ys'].append(ys[indices])

    def shuffle_batch(self, key, batch_size):
        """Draw batches by random shuffling."""
        for indices in batch(
            batch_size, shuffle(key, len(self.memmap['ys']))
        ):
            yield self.memmap['xs'][indices], self.memmap['ys'][indices]


class GDumbCoreset(Coreset):
    """GDumb coreset."""

    def __init__(self, zarr_path, memmap_path, model_spec, coreset_size):
        """Initialize self."""
        super().__init__(zarr_path, memmap_path, model_spec)
        self.coreset_size = coreset_size
        self.zarr['xs'], self.zarr['ys'] = self.empty()

    def update(self, key, xs, ys):
        """Update self."""
        for x, y in zip(xs, ys):
            if len(self.zarr['ys']) < self.coreset_size:
                self.zarr['xs'].append(np.expand_dims(x, 0))
                self.zarr['ys'].append(np.expand_dims(y, 0))
            else:
                key1, key2 = random.split(key)
                ys = self.zarr['ys'][:]
                count = np.bincount(ys)
                mode = random.choice(
                    key1, (count == count.max()).nonzero()[0]
                )
                index = random.choice(key2, (ys == mode).nonzero()[0]).item()
                self.zarr['xs'][index] = x
                self.zarr['ys'][index] = y

    def choice(self, key, batch_size):
        """Draw a batch by random choice."""
        if len(self.memmap['ys']) == 0:
            return self.empty()
        indices = random.choice(
            key, len(self.memmap['ys']), shape=(batch_size,), replace=False
        )
        return self.memmap['xs'][indices], self.memmap['ys'][indices]


class TaskIncrementalCoreset(Coreset):
    """Task-incremental coreset."""

    def __init__(
        self, zarr_path, memmap_path, model_spec, coreset_size_per_task
    ):
        """Initialize self."""
        super().__init__(zarr_path, memmap_path, model_spec)
        self.coreset_size_per_task = coreset_size_per_task
        self.task_count = 0

    def update(self, key, xs, ys):
        """Update self."""
        self.task_count += 1
        name = f'task{self.task_count}'
        self.zarr.create_group(name)
        if len(ys) < self.coreset_size_per_task:
            raise ValueError('dataset size less than coreset size per task')
        indices = random.choice(
            key, len(ys),
            shape=(self.coreset_size_per_task,), replace=False
        )
        self.zarr[name]['xs'] = xs[indices]
        self.zarr[name]['ys'] = ys[indices]

    def choice(self, key, batch_size_per_task):
        """Draw a batch by random choice."""
        if self.task_count == 0:
            return self.empty()
        indices = random.choice(
            key, len(self.memmap['task1/ys']),
            shape=(batch_size_per_task,),
            replace=False
        )
        tree_batch = tree_util.tree_map(lambda x: x[indices], self.memmap)
        return tuple(
            np.concatenate([
                array for path, array in tree_batch.items()
                if path.endswith(s)
            ]) for s in ['xs', 'ys']
        )
