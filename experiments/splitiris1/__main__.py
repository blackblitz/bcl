"""Script for Split Iris 1."""

import jax.numpy as jnp
from jax import flatten_util, vmap
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
from tqdm import tqdm

from dataio.datasets import memmap_dataset
from dataio.dataset_sequences import accumulate_full
from dataio.dataset_sequences.splitsklearn import SplitIris1
from train import Trainer
from train.qc import (
    AutodiffQuadraticConsolidation,
    ElasticWeightConsolidation,
    SynapticIntelligence
)
from train.nc import NeuralConsolidation
from .models import lreg, make_state, nnet

plt.style.use('bmh')


def plot_loss(ax, state, loss, x, y, lim=10.0, vmax=1000.0):
    """Plot loss function."""
    pflat, punflatten = flatten_util.ravel_pytree(state.params)
    grid = np.linspace(-lim, lim, num=200)
    gridx1, gridx2 = jnp.meshgrid(grid, grid)
    gridxs = jnp.vstack([gridx1.ravel(), gridx2.ravel()]).T
    gridys = vmap(
        loss, in_axes=(0, None, None)
    )(vmap(punflatten)(gridxs), x, y)
    gridy = jnp.reshape(gridys, gridx1.shape)
    mesh = ax.pcolormesh(
        gridx1, gridx2, gridy, cmap='inferno', vmin=0.0, vmax=vmax
    )
    ax.plot(*pflat, 'wx')
    return mesh


splitiris1_train = SplitIris1()
splitiris1_test = SplitIris1(train=False)
labels = [
    'Joint training',
    'Fine-tuning',
    'Elastic\nWeight\nConsolidation',
    'Synaptic\nIntelligence',
    'Autodiff\nQuadratic\nConsolidation',
    'Neural\nConsolidation'
]
trainer_kwargs = {
    'batch_size_hyperparams': None, 'batch_size_state': None,
    'multiclass': False, 'n_epochs': 1000
}
for name, model in zip(tqdm(['lreg', 'nnet'], unit='model'), [lreg, nnet]):
    fig = plt.figure(figsize=(12, 6.75))
    axes = ImageGrid(
        fig, 111, nrows_ncols=(3, 6), share_all=True, aspect=False,
        axes_pad=0.25, cbar_location='right', cbar_mode='single',
        cbar_size='5%', cbar_pad=0.25, direction='column'
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
        datasets = tqdm(splitiris1_train, leave=False, unit='task')
        for j, dataset in enumerate(
            accumulate_full(datasets) if i == 0 else datasets
        ):
            x, y = memmap_dataset(dataset)
            trainer.train(x, y)
            mesh = plot_loss(
                axes[j + i * 3], trainer.state, trainer.loss, x, y
            )
            axes[j].set_ylabel(f'Task {j + 1}')
        axes[2 + i * 3].set_xlabel(label)
    axes[0].cax.colorbar(mesh)
    fig.savefig(f'plots/splitiris1_loss_{name}.png')
