"""Script for Split Iris 2."""

from copy import deepcopy
from itertools import accumulate

import jax.numpy as jnp
from jax import flatten_util, random, vmap
from jax.nn import softmax
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from torch.utils.data import ConcatDataset
from tqdm import tqdm

from train import ah, ewc, make_loss_sce, make_step, nc, reg
from evaluate.softmax import accuracy
from torchds import fetch
from torchds.dataset_sequences.splitiris import SplitIris2
from .models import sreg, make_state, nnet

plt.style.use('bmh')


def plot_pred(ax, state):
    gridx1, gridx2 = np.meshgrid(
        jnp.linspace(0.0, 7.0, num=200),
        jnp.linspace(0.0, 3.0, num=200),
    )
    gridxs = np.vstack([gridx1.ravel(), gridx2.ravel()]).T
    gridys = softmax(state.apply_fn({'params': state.params}, gridxs))
    gridy = np.reshape(gridys, (*gridx1.shape, 3))
    return ax.pcolormesh(gridx1, gridx2, gridy, cmap='inferno', vmin=0.0, vmax=1.0)


def plot_xy(ax, x, y):
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
    'Elastic Weight\nConsolidation',
    'Autodiff Quadratic\nConsolidation',
    'Neural Consolidation'
]
algos = [reg, reg, ewc, ah, nc]
hyperparams_inits = [
    {'precision': 0.1},
    {'precision': 0.1},
    {'precision': 0.1, 'lambda': 1.0},
    {'precision': 0.1},
    {'precision': 0.1, 'radius': 20.0, 'size': 100000}
]
for name, model in zip(tqdm(['sreg', 'nnet']), [sreg, nnet]):
    fig, axes = plt.subplots(3, len(labels), sharex=True, sharey=True, figsize=(12, 6.75))
    state_main_init, state_consolidator_init = make_state(model.Main(), model.Consolidator())
    hyperparams_inits[4]['state_consolidator'] = state_consolidator_init
    for i, label in enumerate(tqdm(labels, leave=False, unit='algorithm')):
        hyperparams = deepcopy(hyperparams_inits[i])
        state_main = state_main_init
        loss_basic = make_loss_sce(state_main)
        datasets = tqdm(splitiris2_train, leave=False, unit='task')
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
            plot_pred(axes[j, i], state_main)
            plot_xy(axes[j, i], x, y)
            axes[j, i].set_title(
                'AA: {:.4f}'.format(np.mean([
                    accuracy(state_main,fetch(dataset, 1, None))
                    for dataset in list(splitiris2_test)[: j + 1]
                ]))
            )
            axes[j, 0].set_ylabel(f'Time {j + 1}')
        axes[-1, i].set_xlabel(label)
    fig.tight_layout()
    fig.savefig(f'plots/splitiris2_pred_{name}.png')
