"""Script for Pretrained Split MNIST."""

from copy import deepcopy
from itertools import accumulate
from importlib.resources import files

import jax.numpy as jnp
from jax import jit, random
import matplotlib.pyplot as plt
import numpy as np
from orbax.checkpoint import PyTreeCheckpointer
from torch.utils.data import ConcatDataset
from tqdm import tqdm

from train import ah, ewc, make_loss_sce, make_step, nc, reg
from evaluate.softmax import accuracy
from torchds import fetch
from torchds.dataset_sequences.splitmnist import SplitMNIST
from .models import make_state, sr
from .pretrain.models import make_state_pretrained
from .pretrain.models.cnn import Model

plt.style.use('bmh')

params = PyTreeCheckpointer().restore(
    files('experiments.pretrained_splitmnist.pretrain') / 'cnn'
)
state = make_state_pretrained(Model()).replace(params=params)


@jit
def apply(x):
    return state.apply_fn(
        {"params": state.params}, x,
        method=lambda m, x: m.feature_extractor(x)
    )


splitmnist_train = SplitMNIST()
splitmnist_test = SplitMNIST(train=False)
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
    {'precision': 0.1, 'radius': 30.0, 'size': 10000}
]
markers = 'ovsPX'
for name, model in zip(tqdm(['sr'], unit='model'), [sr]):
    fig, ax = plt.subplots(figsize=(12, 6.75))
    state_main_init, state_consolidator_init = make_state(model.Main(), model.Consolidator())
    hyperparams_inits[4]['state_consolidator'] = state_consolidator_init
    xs = range(1, len(SplitMNIST()) + 1)
    for i, label in enumerate(tqdm(labels, leave=False, unit='algorithm')):
        hyperparams = deepcopy(hyperparams_inits[i])
        state_main = state_main_init
        loss_basic = make_loss_sce(state_main)
        aa = []
        datasets = tqdm(splitmnist_train, leave=False, unit='task')
        if label == 'Joint training':
            datasets = accumulate(datasets, func=lambda x, y: ConcatDataset([x, y]))
        for j, dataset in enumerate(datasets):
            loss = (reg.make_loss if j == 0 else algos[i].make_loss)(
                state_main, hyperparams, loss_basic
            )
            step = make_step(loss)
            for x, y in map(lambda x: (apply(x[0]), x[1]), fetch(dataset, 10, 64)):
                state_main = step(state_main, x, y)
            algos[i].update_hyperparams(
                state_main, hyperparams, loss_basic,
                map(lambda x: (apply(x[0]), x[1]), fetch(dataset, 1, 1024))
            )
            aa.append(
                np.mean([
                    accuracy(
                        state_main,
                        map(
                            lambda x: (apply(x[0]), x[1]),
                            fetch(dataset, 1, 1024)
                        )
                    )
                    for dataset in list(splitmnist_test)[:j + 1]
                ])
            )
        ax.plot(xs, aa, marker=markers[i], markersize=10, alpha=0.5, label=label)
    ax.set_xticks(xs)
    ax.set_ylim([-0.1, 1.1])
    ax.set_xlabel('Time')
    ax.set_ylabel('Average Accuracy')
    ax.legend()
    fig.savefig(f'plots/pretrained_splitmnist_aa_{name}.png')
