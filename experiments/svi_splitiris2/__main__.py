"""Script for Sequential Variational Inference - Split Iris 2."""

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from tqdm import tqdm

from dataio.datasets import to_arrays
from dataio.dataset_sequences.splitsklearn import SplitIris2
from evaluate import predict_proba_softmax_map, predict_proba_softmax_bma
from train import Finetuning
from train.replay import (
    Joint, RandomCoresetReplay, BalancedRandomCoresetReplay
)
from train.svi import (
    GaussianVCL, GaussianMixtureVCL,
    RandomCoresetGaussianSFSVI, BalancedRandomCoresetGaussianSFSVI
)
from .models import nnet3, nnet20, sreg

plt.style.use('bmh')

gridx1, gridx2 = np.meshgrid(
    np.linspace(0.0, 7.0, num=200),
    np.linspace(0.0, 3.0, num=200),
)
gridxs = np.vstack([gridx1.ravel(), gridx2.ravel()]).T


def plot_pred(ax, predict_proba, apply, params):
    """Plot prediction probabilities as pseudo-color plot."""
    gridy = np.reshape(
        predict_proba(apply, params, gridxs),
        (*gridx1.shape, 3)
    )
    return ax.pcolormesh(
        gridx1, gridx2, gridy
    )


def plot_dataset(ax, dataset):
    """Plot x-y pairs as scatter plot."""
    xs, ys = to_arrays(dataset, memmap=False)
    return ax.scatter(
        xs[:, 0], xs[:, 1],
        c=ys, cmap=ListedColormap(list('rgb')),
        vmin=0, vmax=2, s=10.0, linewidths=0.5, edgecolors='w'
    )


splitiris2 = SplitIris2()
hyperparams = {
    'batch_size_hyperparams': None, 'batch_size_state': 16,
    'n_epochs': 100, 'lr': 0.1, 'multiclass': True,
    'input_zero': np.zeros((1, 2)), 'precision': 0.1, 'memmap': False
}
for name, module in zip(
    tqdm(['sreg', 'nnet3', 'nnet20'], unit='model'),
    [sreg, nnet3, nnet20]
):
    model = module.Model()
    fig, axes = plt.subplots(
        3, 8, figsize=(12, 6.75), sharex=True, sharey=True
    )
    trainers = {
        'Joint training': Joint(model, hyperparams),
        'Fine-tuning': Finetuning(model, hyperparams),
        'Random-Coreset\nReplay': RandomCoresetReplay(
            model, hyperparams | {'coreset_size': 20}
        ),
        'Balanced-\nRandom-Coreset\nReplay': BalancedRandomCoresetReplay(
            model, hyperparams | {'coreset_size': 20}
        ),
        'Gaussian VCL': GaussianVCL(
            model, hyperparams | {'sample_size': 1000}
        ),
        'Gaussian-Mixture\nVCL': GaussianMixtureVCL(
            model, hyperparams | {'sample_size': 1000, 'n_comp': 3},
        ),
        'Random-Coreset\nGaussian\nS-FSVI': RandomCoresetGaussianSFSVI(
            model, hyperparams | {'sample_size': 1000, 'coreset_size': 20}
        ),
        'Balanced-\nRandom-Coreset\nGaussian\nS-FSVI':
        BalancedRandomCoresetGaussianSFSVI(
            model, hyperparams | {'sample_size': 1000, 'coreset_size': 20}
        )
    }
    for i, (label, trainer) in enumerate(
        tqdm(trainers.items(), leave=False, unit='algorithm')
    ):
        for j, task in enumerate(tqdm(
            splitiris2, leave=False, unit='task'
        )):
            trainer.train(task)
            plot_pred(
                axes[j, i],
                predict_proba_softmax_map if i < 4
                else predict_proba_softmax_bma,
                trainer.state.apply_fn,
                trainer.state.params if i < 4
                else trainer.sample()
            )
            plot_dataset(axes[j, i], task)
            axes[j, 0].set_ylabel(f'Task {j + 1}')
        axes[-1, i].set_xlabel(label, fontsize=10)
    fig.savefig(f'plots/svi_splitiris2_pred_{name}.png')
