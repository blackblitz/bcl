"""Script for Split Iris 1."""

from copy import deepcopy
from itertools import accumulate

import jax.numpy as jnp
from jax import flatten_util, random, vmap
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
from optax import sigmoid_binary_cross_entropy
from tqdm import tqdm

from train.aqc import AutodiffQuadraticConsolidation
from train.ewc import ElasticWeightConsolidation
from train.loss import make_loss_single_output
from train.nc import NeuralConsolidation
from train.reg import RegularTrainer
from dataio import iter_batches
from dataio.datasets import memmap_dataset
from dataio.dataset_sequences import accumulate_full
from dataio.dataset_sequences.splitiris import SplitIris1
from .models import lreg, make_state, nnet

plt.style.use('bmh')


def plot_loss(ax, state, loss, x, y, lim=10.0, vmax=1000.0):
    pflat, punflatten = flatten_util.ravel_pytree(state.params)
    grid = np.linspace(-lim, lim, num=200)
    gridx1, gridx2 = jnp.meshgrid(grid, grid)
    gridxs = jnp.vstack([gridx1.ravel(), gridx2.ravel()]).T
    gridys = vmap(loss, in_axes=(0, None, None))(vmap(punflatten)(gridxs), x, y)
    gridy = jnp.reshape(gridys, gridx1.shape)
    mesh = ax.pcolormesh(gridx1, gridx2, gridy, cmap='inferno', vmin=0.0, vmax=vmax)
    ax.plot(*pflat, 'wx')
    return mesh


splitiris1_train = SplitIris1()
splitiris1_test = SplitIris1(train=False)
for name, model in zip(tqdm(['lreg', 'nnet'], unit='model'), [lreg, nnet]):
    fig = plt.figure(figsize=(12, 6.75))
    axes = ImageGrid(
        fig, 111, nrows_ncols=(3, 5), share_all=True, aspect=False,
        axes_pad=0.25, cbar_location='right', cbar_mode='single',
        cbar_size='5%', cbar_pad=0.25, direction='column'
    )
    state_main, state_consolidator = make_state(
        model.Main(), model.Consolidator()
    )
    loss_basic = make_loss_single_output(
        state_main, sigmoid_binary_cross_entropy
    )
    labels = tqdm([
        'Joint training',
        'Fine-tuning',
        'Elastic Weight\nConsolidation',
        'Autodiff Quadratic\nConsolidation',
        'Neural Consolidation'
    ], leave=False, unit='algorithm')
    trainers = [
        RegularTrainer(state_main, {'precision': 0.1}, loss_basic),
        RegularTrainer(state_main, {'precision': 0.1}, loss_basic),
        ElasticWeightConsolidation(state_main, {'precision': 0.1, 'lambda': 1.0}, loss_basic),
        AutodiffQuadraticConsolidation(state_main, {'precision': 0.1}, loss_basic),
        NeuralConsolidation(
            state_main,
            {
                'precision': 0.1, 'scale': 10.0, 'size': 1024, 'nsteps': 10000,
                'state_consolidator_init': state_consolidator
            },
            loss_basic
        )
    ]
    for i, (label, trainer) in enumerate(zip(labels, trainers)):
        datasets = tqdm(splitiris1_train, leave=False, unit='task')
        for j, dataset in enumerate(
            accumulate_full(datasets) if i == 0 else datasets
        ):
            x, y = memmap_dataset(dataset)
            trainer.train(1000, None, 1024, x, y)
            mesh = plot_loss(axes[j + i * 3], trainer.state, trainer.loss, x, y)
            axes[j].set_ylabel(f'Time {j + 1}')
        axes[2 + i * 3].set_xlabel(label)
    axes[0].cax.colorbar(mesh)
    fig.savefig(f'plots/splitiris1_loss_{name}.png')
