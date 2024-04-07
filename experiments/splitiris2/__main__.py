"""Script for Split Iris 2."""

from copy import deepcopy
from itertools import islice

import jax.numpy as jnp
from jax import flatten_util, random, vmap
from jax.nn import softmax
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from tqdm import tqdm

from train import make_loss_sce
from train.ah import ah
from train.ewc import ewc
from train.finetune import finetune
from train.joint import joint
from train.nc import nc

from evaluate.softmax import accuracy

from datasets import fetch
from .data import SplitIris2
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


splitiris2 = SplitIris2()
labels = [
    'Joint training',
    'Fine-tuning',
    'Elastic Weight Consolidation',
    'Autodiff Hessian',
    'Neural Consolidation'
]
algos = [joint, finetune, ewc, ah, nc]
hyperparams_inits = [
    {'precision': 0.1},
    {'precision': 0.1},
    {'init': True, 'precision': 0.1, 'lambda': 1.0},
    {'init': True, 'precision': 0.1},
    {'init': True, 'precision': 0.1, 'radius': 20.0, 'size': 10000}
]
for name, model in zip(['sreg', 'nnet'], [sreg, nnet]):
    fig, axes = plt.subplots(3, len(labels), sharex=True, sharey=True, figsize=(12, 6.75))
    state_main_init, state_consolidator_init = make_state(model.Main(), model.Consolidator())
    hyperparams_inits[4]['state_consolidator'] = state_consolidator_init
    for i, label in enumerate(labels):
        hyperparams = deepcopy(hyperparams_inits[i])
        state_main = state_main_init
        for j, dataset_train in enumerate(splitiris2.train()):
            state_main, hyperparams, loss = algos[i](
                make_loss_sce, 1000, None, state_main, hyperparams, dataset_train
            )
            x, y = next(fetch(hyperparams.get('coreset', dataset_train), 1, None))
            plot_pred(axes[j, i], state_main)
            plot_xy(axes[j, i], x, y)
            axes[j, i].set_title(
                'AA: {:.4f}'.format(np.mean([
                    accuracy(None, state_main, dataset_test)
                    for dataset_test in islice(splitiris2.test(), 0, j + 1)
                ]))
            )
            axes[j, 0].set_ylabel(f'Time {j + 1}')
        axes[-1, i].set_xlabel(label)
    fig.tight_layout()
    fig.savefig(f'plots/splitiris2_pred_{name}.png')
