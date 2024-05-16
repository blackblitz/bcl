"""Script for Split Iris 2."""

import jax.numpy as jnp
from jax.nn import softmax
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
from tqdm import tqdm

from dataio.datasets import memmap_dataset
from dataio.dataset_sequences import accumulate_full
from dataio.dataset_sequences.splitsklearn import SplitIris2
from train import Trainer
from train.qc import (
    AutodiffQuadraticConsolidation,
    ElasticWeightConsolidation,
    SynapticIntelligence
)
from train.nc import NeuralConsolidation
from .models import make_state, nnet, sreg

plt.style.use('bmh')


def plot_pred(ax, state):
    """Plot prediction probabilities as pseudo-color plot."""
    ax.set_xticks(range(8))
    ax.set_yticks(range(4))
    gridx1, gridx2 = np.meshgrid(
        jnp.linspace(0.0, 7.0, num=200),
        jnp.linspace(0.0, 3.0, num=200),
    )
    gridxs = np.vstack([gridx1.ravel(), gridx2.ravel()]).T
    gridys = softmax(state.apply_fn({'params': state.params}, gridxs))
    gridy = np.reshape(gridys, (*gridx1.shape, 3))
    return ax.pcolormesh(gridx1, gridx2, gridy, vmin=0.0, vmax=1.0)


def plot_xy(ax, x, y):
    """Plot x-y pairs as scatter plot."""
    return ax.scatter(
        x[:, 0], x[:, 1],
        c=y, cmap=ListedColormap(['r', 'g', 'b']),
        vmin=0, vmax=2, s=10.0, linewidths=0.5, edgecolors='w'
    )


splitiris2_train = SplitIris2()
splitiris2_test = SplitIris2(train=False)
labels = [
    'Joint training',
    'Fine-tuning',
    'Elastic\nWeight\nConsolidation',
    'Synaptic\nIntelligence',
    'Autodiff\nQuadratic\nConsolidation',
    'Neural\nConsolidation'
]
trainer_kwargs = {
    'batch_size_hyperparams': None, 'batch_size_state': None, 'n_epochs': 1000
}
for name, model in zip(tqdm(['sreg', 'nnet']), [sreg, nnet]):
    fig = plt.figure(figsize=(12, 6.75))
    axes = ImageGrid(
        fig, 111, nrows_ncols=(3, 6), share_all=True,
        aspect=False, axes_pad=0.25, direction='column'
    )
    state_main, state_consolidator = make_state(
        model.Main(), model.Consolidator()
    )
    trainers = [
        Trainer(state_main, {'precision': 0.1}, **trainer_kwargs),
        Trainer(state_main, {'precision': 0.1}, **trainer_kwargs),
        ElasticWeightConsolidation(
            state_main, {'precision': 0.1, 'lambda': 1.0}, **trainer_kwargs
        ),
        SynapticIntelligence(
            state_main, {'precision': 0.1, 'xi': 1.0}, **trainer_kwargs
        ),
        AutodiffQuadraticConsolidation(
            state_main, {'precision': 0.1}, **trainer_kwargs
        ),
        NeuralConsolidation(
            state_main,
            {
                'precision': 0.1, 'radius': 10.0, 'size': 1024, 'nsteps': 1000,
                'state': state_consolidator, 'reg': 0.1
            },
            **trainer_kwargs
        )
    ]
    for i, (label, trainer) in enumerate(
        zip(tqdm(labels, leave=False, unit='algorithm'), trainers)
    ):
        datasets = tqdm(splitiris2_train, leave=False, unit='task')
        for j, dataset in enumerate(
            accumulate_full(datasets) if i == 0 else datasets
        ):
            x, y = memmap_dataset(dataset)
            trainer.train(x, y)
            mesh = plot_pred(axes[j + i * 3], trainer.state)
            plot_xy(axes[j + i * 3], x, y)
            axes[j].set_ylabel(f'Task {j + 1}')
        axes[2 + i * 3].set_xlabel(label)
    fig.savefig(f'plots/splitiris2_pred_{name}.png')
