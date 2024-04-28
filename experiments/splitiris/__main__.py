"""Script for Split Iris."""

from copy import deepcopy
from itertools import accumulate

import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import ConcatDataset
from tqdm import tqdm

from train import ah, ewc, make_loss_sce, make_step, nc, reg
from evaluate.softmax import accuracy
from torchds import fetch
from torchds.dataset_sequences.splitiris import SplitIris
from .models import make_state, nnet, sreg

plt.style.use('bmh')

splitiris_train = SplitIris()
splitiris_test = SplitIris(train=False)
labels = [
    'Joint training',
    'Fine-tuning',
    'Elastic Weight Consolidation',
    'Autodiff Quadratic Consolidation',
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
markers = 'ovsPX'
for name, model in zip(tqdm(['sreg', 'nnet'], unit='model'), [sreg, nnet]):
    fig, ax = plt.subplots(figsize=(12, 6.75))
    state_main_init, state_consolidator_init = make_state(model.Main(), model.Consolidator())
    hyperparams_inits[4]['state_consolidator'] = state_consolidator_init
    xs = range(1, len(splitiris_train) + 1)
    for i, label in enumerate(tqdm(labels, leave=False, unit='algorithm')):
        hyperparams = deepcopy(hyperparams_inits[i])
        state_main = state_main_init
        loss_basic = make_loss_sce(state_main)
        aa = []
        datasets = tqdm(splitiris_train, leave=False, unit='task')
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
            aa.append(
                np.mean([
                    accuracy(state_main, fetch(dataset, 1, None))
                    for dataset in list(splitiris_test)[:j + 1]
                ])
            )
        ax.plot(xs, aa, marker=markers[i], markersize=10, alpha=0.5, label=label)
    ax.set_xticks(xs)
    ax.set_ylim([-0.1, 1.1])
    ax.set_xlabel('Time')
    ax.set_ylabel('Average Accuracy')
    ax.legend()
    fig.savefig(f'plots/splitiris_aa_{name}.png')
