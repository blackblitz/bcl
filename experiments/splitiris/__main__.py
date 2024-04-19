"""Script for Split Iris."""

from copy import deepcopy
from itertools import islice

import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from train import make_loss_sce
from train.ah import ah
from train.ewc import ewc
from train.finetune import finetune
from train.joint import joint
from train.nc import nc

from evaluate.softmax import accuracy

from torchds.dataset_sequences.splitiris import SplitIris
from .models import make_state, nnet, sreg

plt.style.use('bmh')

splitiris_train = SplitIris()
splitiris_test = SplitIris(train=False)
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
markers = 'ovsPX'
for name, model in zip(tqdm(['sreg', 'nnet'], unit='model'), [sreg, nnet]):
    fig, ax = plt.subplots(figsize=(12, 6.75))
    state_main_init, state_consolidator_init = make_state(model.Main(), model.Consolidator())
    hyperparams_inits[4]['state_consolidator'] = state_consolidator_init
    for i, label in enumerate(tqdm(labels, unit='algorithm', leave=False)):
        hyperparams = deepcopy(hyperparams_inits[i])
        state_main = state_main_init
        aa = []
        xs = range(1, 4)
        for j, dataset in enumerate(tqdm(
            splitiris_train, unit='task', leave=False
        )):
            state_main, hyperparams, loss = algos[i](
                make_loss_sce, 1000, None, state_main, hyperparams, dataset
            )
            aa.append(
                np.mean([
                    accuracy(None, state_main, dataset)
                    for dataset in list(splitiris_test)[: j + 1]
                ])
            )
        ax.plot(xs, aa, marker=markers[i], markersize=10, alpha=0.5, label=label)
    ax.set_xticks(xs)
    ax.set_ylim([-0.1, 1.1])
    ax.set_xlabel('Time')
    ax.set_ylabel('Average Accuracy')
    ax.legend()
    fig.savefig(f'plots/splitiris_aa_{name}.png')
