"""Script for Split Iris 1."""

from copy import deepcopy
from itertools import accumulate

import jax.numpy as jnp
from jax import flatten_util, random, vmap
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import ConcatDataset
from tqdm import tqdm

from train import ah, ewc, make_loss_bce, make_step, nc, reg
from evaluate.sigmoid import accuracy
from torchds import fetch
from torchds.dataset_sequences.splitiris import SplitIris1
from .models import lreg, make_state, nnet

plt.style.use('bmh')


def plot_loss(ax, state, loss, x, y, lim=20.0, vmax=2000.0):
    pflat, punflatten = flatten_util.ravel_pytree(state.params)
    grid = np.linspace(-lim, lim, num=200)
    gridx1, gridx2 = jnp.meshgrid(grid, grid)
    gridxs = jnp.vstack([gridx1.ravel(), gridx2.ravel()]).T
    gridys = vmap(loss, in_axes=(0, None, None))(vmap(punflatten)(gridxs), x, y)
    gridy = jnp.reshape(gridys, gridx1.shape)
    mesh = ax.pcolormesh(gridx1, gridx2, gridy, cmap='inferno', vmin=0.0, vmax=vmax)
    ax.annotate('*', tuple(pflat), color='w')
    return mesh


splitiris1_train = SplitIris1()
splitiris1_test = SplitIris1(train=False)
labels = [
    'Joint training',
    'Fine-tuning',
    'Elastic Weight Consolidation',
    'Autodiff Hessian',
    'Neural Consolidation'
]
algos = [reg, reg, ewc, ah, nc]
hyperparams_inits = [
    {'precision': 0.1},
    {'precision': 0.1},
    {'precision': 0.1, 'lambda': 1.0},
    {'precision': 0.1},
    {'precision': 0.1, 'radius': 20.0, 'size': 10000}
]
for name, model in zip(tqdm(['lreg', 'nnet'], unit='model'), [lreg, nnet]):
    fig, axes = plt.subplots(3, len(labels), sharex=True, sharey=True, figsize=(12, 6.75))
    state_main_init, state_consolidator_init = make_state(model.Main(), model.Consolidator())
    hyperparams_inits[4]['state_consolidator'] = state_consolidator_init
    for i, label in enumerate(tqdm(labels, leave=False, unit='algorithm')):
        hyperparams = deepcopy(hyperparams_inits[i])
        state_main = state_main_init
        loss_basic = make_loss_bce(state_main)
        datasets = tqdm(splitiris1_train, leave=False, unit='task')
        if label == 'Joint training':
            datasets = accumulate(datasets, func=lambda x, y: ConcatDataset([x, y]))
        for j, dataset in enumerate(datasets):
            loss = (reg.make_loss if j == 0 else algos[i].make_loss)(
                state_main, hyperparams, loss_basic
            )
            step = make_step(loss)
            for x, y in fetch(dataset, 1000, None):
                state_main = step(state_main, x, y)
            algos[i].update_hyperparams(
                state_main, hyperparams, loss_basic, fetch(dataset, 1, None)
            )
            x, y = next(fetch(dataset, 1, None))
            plot_loss(axes[j, i], state_main, loss, x, y)
            axes[j, i].set_title(
                'AA: {:.4f}'.format(np.mean([
                    accuracy(state_main, fetch(dataset, 1, None))
                    for dataset in list(splitiris1_test)[: j + 1]
                ]))
            )
            axes[j, 0].set_ylabel(f'Time {j + 1}')
        axes[-1, i].set_xlabel(label)
    fig.tight_layout()
    fig.savefig(f'plots/splitiris1_loss_{name}.png')
