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
from evaluate import predict
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

    # read experiment specifications and metadata
    exp_path = (Path('experiments') / args.experiment_id).resolve()
    spec = read_toml(exp_path / 'spec.toml')
    ts_path = (Path('data') / spec['task_sequence']['name']).resolve()
    metadata = read_toml(ts_path / 'metadata.toml')

    # create plotter, model, trainers and checkpointer
    plotter = Plotter(
        args.num_classes, args.left, args.right, args.bottom, args.top
    )
    model = getattr(models, spec['model']['name'])(**spec['model']['spec'])
    trainers = [
        (
            trainer['id'],
            getattr(train, trainer['name'])(model, trainer['immutables']),
            partial(
                getattr(predict, predictor['name']),
                **predictor.get('spec', {})
            )
        ) for trainer in spec['trainers']
        for predictor in [trainer['predictor']]
    ]
    ckpter = ocp.StandardCheckpointer()

    # restore checkpoint, predict and plot
    fig, axes = plt.subplots(
        metadata['length'], len(trainers),
        figsize=(12, 6.75), sharex=True, sharey=True
    )
    for i, (trainer_id, trainer, make_predictor) in enumerate(trainers):
        for j, (xs, ys) in enumerate(iter_tasks(ts_path, 'training')):
            params = ckpter.restore(
                exp_path.resolve() / f'ckpt/{trainer_id}_{j + 1}',
                target=trainer.init_state().params
            )
            plotter.plot_pred(axes[j, i], make_predictor(model.apply, params))
            plotter.plot_dataset(axes[j, i], xs, ys)
            if i == 0:
                axes[j, 0].set_ylabel(f'Task {j + 1}')
        axes[-1, i].set_xlabel(trainer_id)
    fig.savefig(exp_path / 'result.png')


if __name__ == '__main__':
    main()
