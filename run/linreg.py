"""Script for LinReg."""

from importlib import import_module
from operator import add

import jax.numpy as jnp
from jax import flatten_util, random, tree_util, vmap
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm

from dataseqs.linreg import LinReg
from train import make_loss_sse
from train.ah import ah
from train.ewc import ewc
from train.finetune import finetune
from train.joint import joint
from train.nc import nc, nc_minibatch

plt.style.use('bmh')


def plot_loss(ax, state, loss, x, y, lim=20.0, vmax=5000.0):
    pflat, punflatten = flatten_util.ravel_pytree(state.params)
    gridx1, gridx2 = jnp.meshgrid(
        jnp.linspace(-lim, lim, num=200),
        jnp.linspace(-lim, lim, num=200),
    )
    gridxs = jnp.vstack([gridx1.ravel(), gridx2.ravel()]).T
    gridys = vmap(loss, in_axes=(0, None, None))(vmap(punflatten)(gridxs), x, y)
    gridy = jnp.reshape(gridys, gridx1.shape)
    mesh = ax.pcolormesh(gridx1, gridx2, gridy, cmap='inferno', vmin=0.0, vmax=vmax)
    ax.annotate('*', tuple(pflat), color='w')
    return mesh


for dataset_name, Dataset in zip(['linreg'], [LinReg]):
    dataset = Dataset()
    for model_name in ['lr']:
        module = import_module(f'models.linreg.{model_name}')
        state_init = module.state_init
        state_consolidator_init = module.state_consolidator_init
        loss_sse = make_loss_sse(state_init)
        
        fig, axes = plt.subplots(3, 4, sharex=True, sharey=True, figsize=(12, 6.75))
        for i, (xlabel, algo, kwargs) in enumerate(zip(
            [
                'Joint Training',
                'Fine-tuning',
                'Elastic Weight\nConsolidation\n($\lambda=1$)',
                'Neural Consolidation\n($n=10000,r=20$)'
            ],
            [joint, finetune, ewc, nc],
            [{}, {}, {}, {'state_consolidator': state_consolidator_init}]
        )):
            for j, (loss, state, x, y) in enumerate(
                algo(1000, state_init, loss_sse, dataset, **kwargs)
            ):
                mesh = plot_loss(axes[j, i], state, loss, x, y)
                if i == 0:
                    axes[j, i].set_ylabel(f'Time {j + 1}')
            axes[-1, i].set_xlabel(xlabel)
        fig.colorbar(mesh, ax=axes.ravel().tolist())
        fig.savefig(f'plots/{dataset_name}_{model_name}_loss.png')
        
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
                algo(1000, state_init, loss_sse, dataset, **kwargs)
            ):
                plot_loss(axes[j, i], state, loss, x, y)
                if i == 0:
                    axes[j, i].set_ylabel(f'Time {j + 1}')
            axes[-1, i].set_xlabel(xlabel)
        fig.colorbar(mesh, ax=axes.ravel().tolist())
        fig.savefig(f'plots/{dataset_name}_{model_name}_loss_ewc.png')
        """
        fig, axes = plt.subplots(3, 4, sharex=True, sharey=True, figsize=(12, 6.75))
        hyperparams = {'state_consolidator': state_consolidator_init}
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
                algo(1000, state_init, loss_sse, dataset, **kwargs)
            ):
                plot_loss(axes[j, i], state, loss, x, y)
                if i == 0:
                    axes[j, i].set_ylabel(f'Time {j + 1}')
            axes[-1, i].set_xlabel(xlabel)
        fig.colorbar(mesh, ax=axes.ravel().tolist())
        fig.savefig(f'plots/{dataset_name}_{model_name}_loss_nc.png')
    
        fig, axes = plt.subplots(3, 4, sharex=True, sharey=True, figsize=(12, 6.75))
        for i, (loss, state, x, y) in enumerate(
            joint(1000, state_init, loss_sse, dataset)
        ):
            plot_loss(axes[i, 0], state, loss, x, y)
            axes[i, 0].set_ylabel(f'Time {j + 1}')
        axes[-1, 0].set_xlabel('Full-batch\nJoint Training')
        for i, size in enumerate([16, 32, 64], start=1):
            for j, (loss, state, x, y) in enumerate(
                nc_minibatch(
                    random.PRNGKey(1337), size, 1000,
                    state_init, loss_sse, dataset, state_consolidator=state_consolidator_init
                )
            ):
                mesh = plot_loss(axes[j, i], state, loss, x, y)
            axes[-1, i].set_xlabel(f'Mini-batch\nNeural Consolidation\nwith Batch Size {size}')
        fig.colorbar(mesh, ax=axes.ravel().tolist())
        fig.savefig(f'plots/{dataset_name}_{model_name}_loss_nc_minibatch.png')
        """