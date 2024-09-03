"""Plotting script."""

import argparse
from functools import partial
from pathlib import Path

from jax import tree_util
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import orbax.checkpoint as ocp

from dataops.io import iter_tasks, read_toml
import models
import train

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

    def plot_pred(self, ax, predict):
        """Plot prediction probabilities as pseudo-color plot."""
        gridy = np.reshape(
            predict(self.gridxs, decide=False),
            self.gridx1.shape if self.n_classes == 2
            else (*self.gridx1.shape, 3)
        )
        return ax.pcolormesh(
            self.gridx1, self.gridx2, gridy, **(
                {'vmin': 0.0, 'vmax': 1.0, 'cmap': 'RdBu'}
                if self.n_classes == 2 else {}
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
    parser.add_argument('experiment_id', help='experiment ID')
    parser.add_argument(
        'num_classes', help='number of classes', type=int, choices=[2, 3]
    )
    parser.add_argument('left', help='lower limit of x-axis', type=float)
    parser.add_argument('right', help='upper limit of x-axis', type=float)
    parser.add_argument('bottom', help='lower limit of y-axis', type=float)
    parser.add_argument('top', help='lower limit of y-axis', type=float)
    args = parser.parse_args()

    # read experiment specifications
    exp_path = Path('experiments').resolve()
    exp = read_toml(exp_path / f'{args.experiment_id}.toml')

    # read metadata
    ts_path = Path('data').resolve() / exp['task_sequence']['name']
    metadata = read_toml(ts_path / 'metadata.toml')

    # set checkpoint path
    ckpt_path = Path('results').resolve() / args.experiment_id / 'ckpt'

    # create plotter, model and trainers
    plotter = Plotter(
        args.num_classes, args.left, args.right, args.bottom, args.top
    )
    model = getattr(models, exp['model']['name'])(**exp['model']['spec'])
    trainers = [
        (
            trainer['id'],
            getattr(train, trainer['name'])(
                model, trainer['immutables'], metadata
            )
        ) for trainer in exp['trainers']
    ]

    # restore checkpoint, predict and plot
    fig, axes = plt.subplots(
        metadata['length'], len(trainers),
        figsize=(12, 6.75), sharex=True, sharey=True
    )
    with ocp.StandardCheckpointer() as ckpter:
        for i, (trainer_id, trainer) in enumerate(trainers):
            for j, (xs, ys) in enumerate(iter_tasks(ts_path, 'training')):
                trainer.state = trainer.state.replace(params=ckpter.restore(
                    ckpt_path / f'{trainer_id}_{j + 1}',
                    target=trainer.init_state().params
                ))
                plotter.plot_pred(axes[j, i], trainer.make_predict())
                plotter.plot_dataset(axes[j, i], xs, ys)
                if i == 0:
                    axes[j, 0].set_ylabel(f'Task {j + 1}')
            axes[-1, i].set_xlabel(trainer_id)

    img_path = Path('results').resolve() / args.experiment_id
    img_path.mkdir(parents=True, exist_ok=True)
    ocp.test_utils.erase_and_create_empty(img_path)
    fig.savefig(img_path / 'plot.png')


if __name__ == '__main__':
    main()
