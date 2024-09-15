"""Coreset."""

from abc import ABC, abstractmethod
from pathlib import Path

from jax import random, tree_util
import numpy as np
import zarr

from dataops.array import batch, shuffle
from dataops.io import clear, get_pass_size


class Coreset(ABC):
    """Abstract base class for a coreset."""

    def __init__(self, zarr_path, memmap_path, immutables, metadata):
        """Initialize self."""
        self.zarr_path = Path(zarr_path)
        self.memmap_path = Path(memmap_path)
        self.immutables = immutables
        self.metadata = metadata
        self.zarr = zarr.open(self.zarr_path, mode='w')
        self.memmap = None

    def empty(self):
        """Return empty arrays xs and ys."""
        return (
            np.empty(
                (0, *self.metadata['input_shape']), dtype=np.float32
            ),
            np.empty((0,), dtype=np.uint8)
        )

    def create_memmap(self):
        """Create memory-mapped files."""
        pass_size = get_pass_size(self.metadata['input_shape'])

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

    @abstractmethod
    def choice(self, key):
        """Draw a batch by random choice."""

    @abstractmethod
    def shuffle_batch(self, key):
        """Draw batches by random shuffling."""


class JointCoreset(Coreset):
    """Joint coreset."""

    def __init__(self, zarr_path, memmap_path, immutables, metadata):
        """Initialize self."""
        super().__init__(zarr_path, memmap_path, immutables, metadata)
        self.zarr['xs'], self.zarr['ys'] = self.empty()

    def update(self, key, xs, ys):
        """Update self."""
        pass_size = get_pass_size(self.metadata['input_shape'])
        for indices in batch(pass_size, np.arange(len(ys))):
            self.zarr['xs'].append(xs[indices])
            self.zarr['ys'].append(ys[indices])

    def choice(self, key):
        """Draw a batch by random choice."""
        indices = random.choice(
            key, len(self.zarr['ys']),
            shape=(self.immutables['batch_size'],), replace=False
        ) if len(self.memmap['ys']) > 0 else []
        return self.memmap['xs'][indices], self.memmap['ys'][indices]

    def shuffle_batch(self, key):
        """Draw batches by random shuffling."""
        for indices in batch(
            self.immutables['batch_size'],
            shuffle(key, len(self.memmap['ys']))
        ):
            yield self.memmap['xs'][indices], self.memmap['ys'][indices]


class GDumbCoreset(Coreset):
    """GDumb coreset."""

    def __init__(self, zarr_path, memmap_path, immutables, metadata):
        """Initialize self."""
        super().__init__(zarr_path, memmap_path, immutables, metadata)
        self.zarr['xs'], self.zarr['ys'] = self.empty()

    def update(self, key, xs, ys):
        """Update self."""
        for x, y in zip(xs, ys):
            if len(self.zarr['ys']) < self.immutables['coreset_size']:
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

    def choice(self, key):
        """Draw a batch by random choice."""
        if len(self.memmap['ys']) == 0:
            return self.empty()
        indices = random.choice(
            key, len(self.memmap['ys']),
            shape=(self.immutables['coreset_batch_size'],), replace=False
        )
        return self.memmap['xs'][indices], self.memmap['ys'][indices]

    def shuffle_batch(self, key):
        """Draw batches by random shuffling."""
        if len(self.memmap['ys']) == 0:
            yield self.empty()
        else:
            for indices in batch(
                self.immutables['coreset_batch_size'],
                shuffle(key, len(self.memmap['ys']))
            ):
                yield self.memmap['xs'][indices], self.memmap['ys'][indices]


class TaskIncrementalCoreset(Coreset):
    """Task-incremental coreset."""

    def __init__(self, zarr_path, memmap_path, immutables, metadata):
        """Initialize self."""
        super().__init__(zarr_path, memmap_path, immutables, metadata)
        self.task_count = 0

    def update(self, key, xs, ys):
        """Update self."""
        self.task_count += 1
        name = f'task{self.task_count}'
        self.zarr.create_group(name)
        if len(ys) < self.immutables['coreset_size_per_task']:
            raise ValueError('dataset size less than coreset size per task')
        indices = random.choice(
            key, len(ys),
            shape=(self.immutables['coreset_size_per_task'],), replace=False
        )
        self.zarr[name]['xs'] = xs[indices]
        self.zarr[name]['ys'] = ys[indices]

    def noise(self, key):
        """Return random data."""
        key1, key2 = random.split(key)
        return (
            np.asarray(random.uniform(
                key1,
                shape=(
                    self.immutables['coreset_batch_size_per_task'],
                    *self.metadata['input_shape']
                ),
                minval=self.immutables['noise_minval'],
                maxval=self.immutables['noise_maxval']
            )),
            np.asarray(random.choice(
                key2, len(self.metadata['classes']),
                shape=(self.immutables['coreset_batch_size_per_task'],)
            ))
        )

    def choice(self, key):
        """Draw a batch by random choice."""
        if self.task_count == 0:
            if self.immutables['noise_init']:
                return self.noise(key)
            return self.empty()
        indices = random.choice(
            key, len(self.memmap['task1/ys']),
            shape=(self.immutables['coreset_batch_size_per_task'],),
            replace=False
        )
        tree_batch = tree_util.tree_map(lambda x: x[indices], self.memmap)
        return tuple(
            np.concatenate([
                array for path, array in tree_batch.items()
                if path.endswith(s)
            ]) for s in ['xs', 'ys']
        )

    def shuffle_batch(self, key):
        """Draw batches by random shuffling."""
        if self.task_count == 0:
            if self.immutables['noise_init']:
                yield self.noise(key)
            else:
                yield self.empty()
        else:
            for indices in batch(
                self.immutables['coreset_batch_size_per_task'],
                shuffle(key, len(self.memmap['task1/ys']))
            ):
                tree_batch = tree_util.tree_map(
                    lambda x: x[indices], self.memmap
                )
                yield tuple(
                    np.concatenate([
                        array for path, array in tree_batch.items()
                        if path.endswith(s)
                    ]) for s in ['xs', 'ys']
                )
