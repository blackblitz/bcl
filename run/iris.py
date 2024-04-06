"""Script for Iris."""

from importlib import import_module

import jax.numpy as jnp
from jax.nn import softmax
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm

from dataseqs.iris import PermutedIris, SplitIris
from evaluate import aa_softmax
from train import make_loss_sce
from train.ah import ah
from train.ewc import ewc
from train.finetune import finetune
from train.joint import joint
from train.nc import nc

plt.style.use('bmh')

for dataseq_name, Dataseq in zip(
    ['permutediris', 'splitiris'],
    [PermutedIris, SplitIris]
):
    dataseq = Dataseq()
    for model_name in ['sr', 'nn']:
        module = import_module(f'models.iris.{model_name}')
        state_init = module.state_init
        state_consolidator_init = module.state_consolidator_init
    
        loss_sce = make_loss_sce(state_init)
    
        fig, ax = plt.subplots(figsize=(12, 6.75))
    
        time = jnp.arange(1, 4)
        for i, (label, algo, kwargs) in enumerate(zip(
            tqdm([
                'Joint Training',
                'Fine-tuning',
                'Elastic Weight\nConsolidation\n($\lambda=1$)',
                'Autodiff Hessian',
                'Neural Consolidation\n($n=10000,r=20$)'
            ]),
            [joint, finetune, ewc, ah, nc],
            [{}, {}, {}, {}, {'state_consolidator': state_consolidator_init}]
        )):
            ax.plot(
                time,
                jnp.array([
                    aa_softmax(i, state, dataseq)
                    for i, (state, loss, dataset) in enumerate(
                        algo(1000, state_init, loss_sce, dataseq, **kwargs)
                    )
                ]),
                alpha=0.6,
                label=label
            )
        ax.set_ylim([-0.1, 1.1])
        ax.set_xticks(time)
        ax.set_xlabel('Time')
        ax.set_ylabel('Average Accuracy')
        ax.legend()
    
        fig.savefig(f'plots/{dataseq_name}_{model_name}_aa.png')
