"""Script for Iris2."""

from importlib import import_module

import jax.numpy as jnp
from jax.nn import softmax
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader

from dataseqs import numpy_collate
from dataseqs.iris import SplitIris2
from evaluate import aa_softmax
from train import make_loss_sce
from train.ah import ah
from train.ewc import ewc
from train.finetune import finetune
from train.joint import joint
from train.nc import nc

plt.style.use('bmh')


def plot_pred_softmax(ax, state):
    gridx1, gridx2 = jnp.meshgrid(
        jnp.linspace(0.0, 7.0, num=200),
        jnp.linspace(0.0, 3.0, num=200),
    )
    gridxs = jnp.vstack([gridx1.ravel(), gridx2.ravel()]).T
    gridys = softmax(state.apply_fn({'params': state.params}, gridxs))
    gridy = jnp.reshape(gridys, (*gridx1.shape, -1))
    return ax.pcolormesh(gridx1, gridx2, gridy, cmap='inferno', vmin=0.0, vmax=1.0)


def plot_xy(ax, x, y):
    return ax.scatter(
        x[:, 0], x[:, 1],
        c=y, cmap=ListedColormap(['r', 'g', 'b']),
        vmin=0, vmax=2, s=10.0, linewidths=0.5, edgecolors='w'
    )


dataseq_name = 'splitiris2'
dataseq = SplitIris2()
for model_name in ['sr', 'nn']:
    module = import_module(f'models.iris2.{model_name}')
    state_init = module.state_init
    state_consolidator_init = module.state_consolidator_init
    loss_sce = make_loss_sce(state_init)

    fig, axes = plt.subplots(3, 5, sharex=True, sharey=True, figsize=(12, 6.75))
    for i, (xlabel, algo, kwargs) in enumerate(zip(
        [
            'Joint Training',
            'Fine-tuning',
            'Elastic Weight\nConsolidation\n($\lambda=1$)',
            'Autodiff\nHessian',
            'Neural Consolidation\n($n=10000,r=20$)'
        ],
        [joint, finetune, ewc, ah, nc],
        [{}, {}, {}, {}, {'state_consolidator': state_consolidator_init}]
    )):
        for j, (state, loss, dataset) in enumerate(
            algo(1000, state_init, loss_sce, dataseq, **kwargs)
        ):
            x, y = next(iter(
                DataLoader(
                    dataset,
                    batch_size=len(dataset),
                    collate_fn=numpy_collate
                )
            ))
            plot_pred_softmax(axes[j, i], state)
            scatter = plot_xy(axes[j, i], x, y)
            axes[j, i].set_title(f'AA: {aa_softmax(j, state, dataseq):.4f}')
            if i == 0:
                axes[j, i].set_ylabel(f'Time {j + 1}')
        axes[-1, i].set_xlabel(xlabel)
    fig.savefig(f'plots/{dataseq_name}_{model_name}_pred.png')
    """
    fig, axes = plt.subplots(3, 6, sharex=True, sharey=True, figsize=(12, 6.75))
    for i, (xlabel, algo, kwargs) in enumerate(zip(
        [
            'Joint Training',
            'Autodiff\nHessian',
            'Autodiff\nHessian\nDiagonal',
            'Elastic Weight\nConsolidation\n($\lambda=0.1$)',
            'Elastic Weight\nConsolidation\n($\lambda=1$)',
            'Elastic Weight\nConsolidation\n($\lambda=10$)'
        ],
        [joint, ah, ah, ewc, ewc, ewc],
        [
            {}, {}, {'diagonal': True},
            {'lambda_': 0.1}, {'lambda_': 1.0}, {'lambda_': 10.0}
        ]
    )):
        for j, (loss, state, x, y) in enumerate(
            algo(1000, state_init, loss_sce, dataseq, **kwargs)
        ):
            plot_pred_softmax(axes[j, i], state)
            plot_xy(axes[j, i], x, y)
            axes[j, i].set_title(f'AA: {aa_softmax(j, state, dataseq):.4f}')
            if i == 0:
                axes[j, i].set_ylabel(f'Time {j + 1}')
        axes[-1, i].set_xlabel(xlabel)
    fig.savefig(f'plots/{dataseq_name}_{model_name}_pred_ewc.png')

    fig, axes = plt.subplots(3, 4, sharex=True, sharey=True, figsize=(12, 6.75))
    for i, (xlabel, algo, kwargs) in enumerate(zip(
        [
            'Joint Training',
            'Neural Consolidation\n($n=10000,r=10$)',
            'Neural Consolidation\n($n=10000,r=20$)',
            'Neural Consolidation\n($n=10000,r=30$)'
        ],
        [joint, nc, nc, nc],
        [
            {},
            {
                'state_consolidator': state_consolidator_init.replace(
                    hyperparams=state_consolidator_init.hyperparams | {'radius': 10.0}
                )
            },
            {
                'state_consolidator': state_consolidator_init.replace(
                    hyperparams=state_consolidator_init.hyperparams | {'radius': 20.0}
                )
            },
            {
                'state_consolidator': state_consolidator_init.replace(
                    hyperparams=state_consolidator_init.hyperparams | {'radius': 30.0}
                )
            }
        ]
    )):
        for j, (loss, state, x, y) in enumerate(
            algo(1000, state_init, loss_sce, dataseq, **kwargs)
        ):
            plot_pred_softmax(axes[j, i], state)
            plot_xy(axes[j, i], x, y)
            axes[j, i].set_title(f'AA: {aa_softmax(j, state, dataseq):.4f}')
            if i == 0:
                axes[j, i].set_ylabel(f'Time {j + 1}')
        axes[-1, i].set_xlabel(xlabel)
    fig.savefig(f'plots/{dataseq_name}_{model_name}_pred_nc.png')
    """
