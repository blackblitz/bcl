"""Plotting script."""

import argparse
from importlib import import_module
from multiprocessing import cpu_count
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator
import numpy as np

from .. import Experiment
from ..dataops.io import read_task, read_toml
from ..models import ModelSpec, NLL
from ..models import module_map as models_module_map
from ..train import select_trainers
from ..train.trainer import module_map as trainer_module_map

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
            predictor(self.gridxs, decide=False),
            self.gridx1.shape if self.n_classes == 2
            else (*self.gridx1.shape, 3)
        )
        return ax.pcolormesh(
            self.gridx1, self.gridx2, gridy, **(
                {'vmin': 0.0, 'vmax': 1.0, 'cmap': 'RdBu'}
                if self.n_classes == 2 else {}
            )
        )

    def plot_entr(self, ax, predictor):
        """Plot prediction entropy as pseudo-color plot."""
        gridy = np.reshape(predictor.entropy(self.gridxs), self.gridx1.shape)
        return ax.pcolormesh(
            self.gridx1, self.gridx2, gridy, **(
                {'vmin': 0.0, 'vmax': 1.0, 'cmap': 'Greys'}
            )
        )

    def plot_dataset(self, ax, xs, ys):
        """Plot x-y pairs as scatter plot."""
        ax.scatter(
            xs[:, 0], xs[:, 1], c=ys,
            cmap=(
                ListedColormap(list('rgb')) if self.n_classes == 3
                else ListedColormap(list('rb'))
            ), vmin=0, vmax=self.n_classes,
            s=10.0, linewidths=0.5, edgecolors='w'
        )


def main():
    """Run the main script."""
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_id', help='experiment ID')
    args = parser.parse_args()

    exp = Experiment(args.exp_id)
    exp.setup(is_eval=True)

    plotter = Plotter(
        exp.spec['plot']['n_classes'],
        exp.spec['plot']['x1_min'],
        exp.spec['plot']['x1_max'],
        exp.spec['plot']['x2_min'],
        exp.spec['plot']['x2_max'],
    )

    # restore checkpoint, predict and plot
    if (exp.paths['result'] / 'ckpt/hmcnuts_1').exists():
        exp.spec['trainers'].append({
            'id': 'hmcnuts',
            'label': 'Joint\nHMC-NUTS',
            'name': 'HMCNUTS',
            'hparams': {'predict': {'sample_size': cpu_count()}}
        })
    trainer_spec_map = select_trainers(exp)
    fig, axes = plt.subplots(
        exp.metadata['length'], len(trainer_spec_map),
        figsize=(exp.spec['plot']['width'], exp.spec['plot']['height']),
        sharex=True, sharey=True,
        constrained_layout=True
    )
    axes = np.array([axes])
    axes = axes.reshape((exp.metadata['length'], len(trainer_spec_map)))
    for i, trainer_spec in enumerate(trainer_spec_map.values()):
        trainer_id = trainer_spec['id']
        trainer_label = trainer_spec['label']
        trainer_class = getattr(
            import_module(trainer_module_map[trainer_spec['name']]),
            trainer_spec['name']
        )
        hparams = trainer_spec['hparams']['predict']

        for j in range(exp.metadata['length']):
            task_id = j + 1
            xs, ys = read_task(exp.paths['data'], 'training', task_id)
            path = exp.paths['result'] / f'ckpt/{trainer_id}_{task_id}'
            predictor = trainer_class.predictor_class.from_checkpoint(
                exp.model, exp.mspec, hparams, path
            )
            plotter.plot_pred(axes[j, i], predictor)
            plotter.plot_dataset(axes[j, i], xs, ys)
            if i == 0 and exp.metadata['length'] > 1:
                axes[j, 0].set_ylabel(f'Task {task_id}')
            axes[j, i].xaxis.set_major_locator(MaxNLocator(integer=True))
            axes[j, i].yaxis.set_major_locator(MaxNLocator(integer=True))
        axes[-1, i].set_xlabel(trainer_label)

    (exp.paths['result'] / 'plots').mkdir(parents=True, exist_ok=True)
    fig.savefig(exp.paths['result'] / 'plots/pred.png')


if __name__ == '__main__':
    main()
