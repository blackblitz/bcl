"""Script for Split Iris 1 - size."""

import jax.numpy as jnp
from jax import flatten_util, vmap
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
from tqdm import tqdm

from dataio.datasets import memmap_dataset
from dataio.dataset_sequences import accumulate_full
from dataio.dataset_sequences.splitiris import SplitIris1
from train import Trainer
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
trainer_kwargs = {
    'batch_size_hyperparams': None, 'batch_size_state': None,
    'multiclass': False, 'n_epochs': 1000
}
fig = plt.figure(figsize=(12, 6.75))
axes = ImageGrid(
    fig, 111, nrows_ncols=(2, 6), share_all=True, aspect=False,
    axes_pad=0.25, cbar_location='right', cbar_mode='single',
    cbar_size='5%', cbar_pad=0.25, direction='column'
)
for i, model in enumerate(tqdm([lreg, nnet], unit='model')):
    state_main, state_consolidator = make_state(
        model.Main(), model.Consolidator()
    )
    trainers = [
        Trainer(state_main, {'precision': 0.1}, **trainer_kwargs),
        *(NeuralConsolidation(
            state_main,
            {
                'precision': 0.1, 'radius': 10.0, 'size': size, 'nsteps': 1000,
                'state': state_consolidator, 'reg': 0.1
            },
            **trainer_kwargs
        ) for size in [4, 16, 64, 256, 1024])
    ]
    for j, trainer in enumerate(
        tqdm(trainers, leave=False, unit='algorithm')
    ):
        datasets = tqdm(splitiris1_train, leave=False, unit='task')
        for dataset in (accumulate_full(datasets) if j == 0 else datasets):
            x, y = memmap_dataset(dataset)
            trainer.train(x, y)
        mesh = plot_loss(
            axes[j * 2 + i], trainer.state, trainer.loss, x, y
        )
        if j > 0:
            axes[j * 2 + 1].set_xlabel(
                '$n={}$'.format(trainer.hyperparams.get('size'))
            )
axes[1].set_xlabel('Joint training')
axes[0].cax.colorbar(mesh)
fig.savefig('plots/splitiris1_loss_size.png')
