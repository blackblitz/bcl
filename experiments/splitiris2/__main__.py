"""Script for Split Iris 2."""

import jax.numpy as jnp
from jax import flatten_util, random, vmap
from jax.nn import softmax
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
from optax import softmax_cross_entropy_with_integer_labels
from tqdm import tqdm

from train.aqc import AutodiffQuadraticConsolidation
from train.ewc import ElasticWeightConsolidation
from train.loss import make_loss_multi_output
from train.nc import NeuralConsolidation
from train.reg import RegularTrainer
from dataio.datasets import memmap_dataset
from dataio.dataset_sequences import accumulate_full
from dataio.dataset_sequences.splitiris import SplitIris2
from .models import make_state, nnet, sreg

plt.style.use('bmh')


def plot_pred(ax, state):
    ax.set_xticks(range(8))
    ax.set_yticks(range(4))
    gridx1, gridx2 = np.meshgrid(
        jnp.linspace(0.0, 7.0, num=200),
        jnp.linspace(0.0, 3.0, num=200),
    )
    gridxs = np.vstack([gridx1.ravel(), gridx2.ravel()]).T
    gridys = softmax(state.apply_fn({'params': state.params}, gridxs))
    gridy = np.reshape(gridys, (*gridx1.shape, 3))
    return ax.pcolormesh(gridx1, gridx2, gridy, vmin=0.0, vmax=1.0)


def plot_xy(ax, x, y):
    return ax.scatter(
        x[:, 0], x[:, 1],
        c=y, cmap=ListedColormap(['r', 'g', 'b']),
        vmin=0, vmax=2, s=10.0, linewidths=0.5, edgecolors='w'
    )


splitiris2_train = SplitIris2()
splitiris2_test = SplitIris2(train=False)
for name, model in zip(tqdm(['sreg', 'nnet']), [sreg, nnet]):
    fig = plt.figure(figsize=(12, 6.75))
    axes = ImageGrid(
        fig, 111, nrows_ncols=(3, 5), share_all=True,
        aspect=False, axes_pad=0.25, direction='column'
    )
    state_main, state_consolidator = make_state(
        model.Main(), model.Consolidator()
    )
    loss_basic = make_loss_multi_output(
        state_main, softmax_cross_entropy_with_integer_labels
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
        ElasticWeightConsolidation(
            state_main, {'precision': 0.1, 'lambda': 1.0}, loss_basic
        ),
        AutodiffQuadraticConsolidation(
            state_main, {'precision': 0.1}, loss_basic
        ),
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
        datasets = tqdm(splitiris2_train, leave=False, unit='task')
        for j, dataset in enumerate(
            accumulate_full(datasets) if i == 0 else datasets
        ):
            x, y = memmap_dataset(dataset)
            trainer.train(1000, None, 1024, x, y)
            mesh = plot_pred(axes[j + i * 3], trainer.state)
            plot_xy(axes[j + i * 3], x, y)
            axes[j].set_ylabel(f'Time {j + 1}')
        axes[2 + i * 3].set_xlabel(label)
    fig.savefig(f'plots/splitiris2_pred_{name}.png')
