"""Plotting script."""

import argparse
from importlib import import_module
from multiprocessing import cpu_count
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from dataops.io import read_task, read_toml
from models import ModelSpec, NLL
from models import module_map as models_module_map
from train.trainer import module_map as trainer_module_map

from evaluate import metrics

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
    parser.add_argument('experiment_id', help='experiment ID')
    args = parser.parse_args()

    # read experiment specifications
    exp_path = Path('experiments').resolve()
    exp = read_toml(exp_path / f'{args.experiment_id}.toml')

    # read metadata
    ts_path = Path('data').resolve() / exp['task_sequence']['name']
    metadata = read_toml(ts_path / 'metadata.toml')
    if 'ood_name' in exp['task_sequence']:
        if metadata['length'] > 1 or 'feature_extractor' in exp:
            raise ValueError(
                'OOD testing is only for singleton task sequences '
                'without a pre-trained feature extractor'
            )
        ood_ts_path = Path('data').resolve() / exp['task_sequence']['ood_name']

    # set results path
    results_path = Path('results').resolve() / args.experiment_id

    # create plotter and model
    plotter = Plotter(**exp['plot'])
    model = getattr(
        import_module(models_module_map[exp['model']['name']]),
        exp['model']['name']
    )(**exp['model']['args'])
    mspec = ModelSpec(
        nll=NLL[exp['model']['spec']['nll']],
        in_shape=exp['model']['spec']['in_shape'],
        out_shape=exp['model']['spec']['out_shape']
    )

    # restore checkpoint, predict and plot
    if (results_path / f'ckpt/hmcnuts_1').exists():
        exp['trainers'].append({
            'id': 'hmcnuts',
            'label': 'HMC-NUTS',
            'name': 'HMCNUTS',
            'hparams': {'predict': {'sample_size': cpu_count()}}
        })
    fig, axes = plt.subplots(
        metadata['length'], len(exp['trainers']),
        figsize=(12, 6.75), sharex=True, sharey=True,
        constrained_layout=True
    )
    axes = np.array([axes])
    axes = axes.reshape((metadata['length'], len(exp['trainers'])))
    for i, trainer_spec in enumerate(exp['trainers']):
        trainer_id = trainer_spec['id']
        trainer_label = trainer_spec['label']
        trainer_class = getattr(
            import_module(trainer_module_map[trainer_spec['name']]),
            trainer_spec['name']
        )
        hparams = trainer_spec['hparams']['predict']

        for j in range(metadata['length']):
            task_id = j + 1
            xs, ys = read_task(ts_path, 'training', task_id)
            path = results_path / f'ckpt/{trainer_id}_{task_id}'
            predictor = trainer_class.predictor_class.from_checkpoint(
                model, mspec, hparams, path
            )
            plotter.plot_pred(axes[j, i], predictor)
            plotter.plot_dataset(axes[j, i], xs, ys)
            if i == 0 and metadata['length'] > 1:
                axes[j, 0].set_ylabel(f'Task {task_id}')
        axes[-1, i].set_xlabel(trainer_label)

    (results_path / 'plots').mkdir(parents=True, exist_ok=True)
    fig.savefig(results_path / 'plots/pred.png')
    fig, axes = plt.subplots(
        metadata['length'], len(exp['trainers']),
        figsize=(12, 6.75), sharex=True, sharey=True,
        constrained_layout=True
    )
    axes = np.array([axes])
    axes = axes.reshape((metadata['length'], len(exp['trainers'])))
    for i, trainer_spec in enumerate(exp['trainers']):
        trainer_id = trainer_spec['id']
        trainer_label = trainer_spec['label']
        trainer_class = getattr(
            import_module(trainer_module_map[trainer_spec['name']]),
            trainer_spec['name']
        )
        hparams = trainer_spec['hparams']['predict']

        for j in range(metadata['length']):
            task_id = j + 1
            xs, ys = read_task(ood_ts_path, 'training', task_id)
            path = results_path / f'ckpt/{trainer_id}_{task_id}'
            predictor = trainer_class.predictor_class.from_checkpoint(
                model, mspec, hparams, path
            )
            plotter.plot_entr(axes[j, i], predictor)
            plotter.plot_dataset(axes[j, i], xs, ys)
            if i == 0 and metadata['length'] > 1:
                axes[j, 0].set_ylabel(f'Task {task_id}')
        axes[-1, i].set_xlabel(trainer_label)

    (results_path / 'plots').mkdir(parents=True, exist_ok=True)
    fig.savefig(results_path / 'plots/entr.png')


if __name__ == '__main__':
    main()
