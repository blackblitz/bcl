"""Plotting script."""

import argparse
from functools import partial
from pathlib import Path
from importlib import import_module
import tomllib

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from orbax.checkpoint import PyTreeCheckpointer

from dataio.dataset_sequences.datasets import dataset_to_arrays

plt.style.use('bmh')


class Plotter:
    """Plotter."""

    def __init__(self, n_classes, x1_min, x1_max, x2_min, x2_max):
        """Initialize self."""
        if n_classes in [2, 3]:
            self.n_classes = n_classes
        else:
            raise ValueError('number of classes is not 2 or 3')
        self.gridx1, self.gridx2 = np.meshgrid(
            np.linspace(x1_min, x1_max, num=200),
            np.linspace(x2_min, x2_max, num=200),
        )
        self.gridxs = np.vstack([self.gridx1.ravel(), self.gridx2.ravel()]).T

    def plot_pred(self, ax, predictor):
        """Plot prediction probabilities as pseudo-color plot."""
        gridy = np.reshape(
            predictor.predict_proba(self.gridxs),
            self.gridx1.shape if self.n_classes == 2
            else (*self.gridx1.shape, 3)
        )
        return ax.pcolormesh(
            self.gridx1, self.gridx2, gridy, **(
                {'vmin': 0.0, 'vmax': 1.0, 'cmap': 'RdBu'}
                if self.n_classes == 2 else {}
            )
        )

    def plot_dataset(self, ax, dataset):
        """Plot x-y pairs as scatter plot."""
        xs, ys = dataset_to_arrays(dataset, memmap=False)
        if self.n_classes == 2:
            cmap = ListedColormap(list('rb'))
        return ax.scatter(
            xs[:, 0], xs[:, 1], c=ys,
            cmap=(
                ListedColormap(list('rgb')) if self.n_classes == 3
                else ListedColormap(list('rb'))
            ), vmin=0, vmax=self.n_classes,
            s=10.0, linewidths=0.5, edgecolors='w'
        )


parser = argparse.ArgumentParser()
parser.add_argument('experiment_id')
args = parser.parse_args()

path = Path('experiments') / args.experiment_id
with open(path / 'spec.toml', 'rb') as file:
    spec = tomllib.load(file)

plotter = Plotter(**spec['plotter'])

dataset_sequence = getattr(
    import_module('dataio.dataset_sequences'),
    spec['dataset_sequence']['name']
)(**spec['dataset_sequence']['spec'])['training']
model = getattr(
    import_module(spec['model']['module']),
    spec['model']['name']
)(**spec['model']['spec'])
make_predictors = [
    (
        trainer['id'],
        partial(
            getattr(import_module(predictor['module']), predictor['name']),
            **predictor.get('spec', {})
        )
    ) for trainer in spec['trainers']
    for predictor in [trainer['predictor']]
]
ckpter = PyTreeCheckpointer()
fig, axes = plt.subplots(
    len(dataset_sequence), len(make_predictors),
    figsize=(12, 6.75), sharex=True, sharey=True
)
if len(dataset_sequence) == 1:
    axes = np.expand_dims(axes, 0)
for i, (trainer_id, make_predictor) in enumerate(make_predictors):
    for j, (task_id, task) in enumerate(enumerate(dataset_sequence, start=1)):
        params = ckpter.restore(
            path.resolve() / f'ckpt/{trainer_id}_{task_id}'
        )
        plotter.plot_pred(axes[j, i], make_predictor(model.apply, params))
        plotter.plot_dataset(axes[j, i], task)
        if i == 0:
            axes[j, 0].set_ylabel(f'Task {task_id}')
    axes[-1, i].set_xlabel(trainer_id)
fig.savefig(path / 'result.png')
