"""Script for Split MNIST."""

from copy import deepcopy

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

from .data import SplitMNIST
from .models import make_state, cnn, fcnn

plt.style.use('bmh')

splitmnist_train = SplitMNIST()
splitmnist_test = SplitMNIST(train=False)
labels = [
    'Joint training',
    'Fine-tuning',
    'Elastic Weight Consolidation',
    'Neural Consolidation'
]
algos = [joint, finetune, ewc, nc]
hyperparams_inits = [
    {'precision': 0.1},
    {'precision': 0.1},
    {'init': True, 'precision': 0.1, 'lambda': 1.0},
    {'init': True, 'precision': 0.1, 'radius': 20.0, 'size': 100}
]
markers = 'ovsP'
for name, model in zip(tqdm(['fcnn', 'cnn'], unit='model'), [fcnn, cnn]):
    fig, ax = plt.subplots(figsize=(12, 6.75))
    state_main_init, state_consolidator_init = make_state(model.Main(), model.Consolidator())
    hyperparams_inits[3]['state_consolidator'] = state_consolidator_init
    xs = range(1, len(SplitMNIST()) + 1)
    for i, label in enumerate(tqdm(labels, leave=False, unit='algorithm')):
        hyperparams = deepcopy(hyperparams_inits[i])
        state_main = state_main_init
        aa = []
        for j, dataset_train in enumerate(tqdm(splitmnist_train, leave=False)):
            state_main, hyperparams, loss = algos[i](
                make_loss_sce, 10, 64, state_main, hyperparams, dataset_train
            )
            aa.append(
                np.mean([
                    accuracy(1024, state_main, dataset_test)
                    for dataset_test in list(splitmnist_test)[:j + 1]
                ])
            )
        ax.plot(xs, aa, marker=markers[i], markersize=10, alpha=0.5, label=label)
    ax.set_xticks(xs)
    ax.set_ylim([-0.1, 1.1])
    ax.set_xlabel('Time')
    ax.set_ylabel('Average Accuracy')
    ax.legend()
    fig.savefig(f'plots/splitmnist_aa_{name}.png')
