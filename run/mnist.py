"""Script for MNIST."""

from importlib import import_module

import jax.numpy as jnp
from jax import random
from jax.nn import softmax
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm

from dataseqs.mnist import PermutedMNIST, SplitMNIST
from evaluate import aa_sigmoid, aa_softmax
from train import make_loss_bce, make_loss_sce
from train.ewc import ewc_minibatch
from train.finetune import finetune_minibatch
from train.joint import joint_minibatch
from train.nc import nc_minibatch

plt.style.use('bmh')

key = random.PRNGKey(1337)

for dataset_name, Dataset, binary, model_names, make_loss, aa in zip(
    ['permutedmnist', 'splitmnistci', 'splitmnistdi'],
    [PermutedMNIST, SplitMNIST, SplitMNIST],
    [False, False, True],
    [['cnn10', 'nn10'], ['cnn10', 'nn10'], ['cnn1', 'nn1']],
    [make_loss_sce, make_loss_sce, make_loss_bce],
    [aa_softmax, aa_softmax, aa_sigmoid]
):
    for model_name in model_names:
        dataset = Dataset(binary=binary)
        module = import_module(f'models.mnist.{model_name}')
        state_init = module.state_init
        state_consolidator_init = module.state_consolidator_init
    
        loss_basic = make_loss(state_init) 
    
        fig, ax = plt.subplots(figsize=(12, 6.75))
    
        time = jnp.arange(1, 6)
        for i, (label, algo, kwargs) in enumerate(zip(
            tqdm([
                'Joint Training',
                'Fine-tuning',
                'Elastic Weight\nConsolidation\n($\lambda=1$)',
                'Neural Consolidation\n($n=1000,r=20$)'
            ]),
            [joint_minibatch, finetune_minibatch, ewc_minibatch, nc_minibatch],
            [{}, {}, {}, {'state_consolidator': state_consolidator_init}]
        )):
            ax.plot(
                time,
                jnp.array([
                    aa(i, state, dataset)
                    for i, (loss, state, x, y) in enumerate(
                        algo(key, 64, 10, state_init, loss_basic, dataset, **kwargs)
                    )
                ]),
                alpha=0.8,
                label=label
            )
        ax.set_ylim([-0.1, 1.1])
        ax.set_xticks(time)
        ax.set_xlabel('Time')
        ax.set_ylabel('Average Accuracy')
        ax.legend()
    
        fig.savefig(f'plots/{dataset_name}_{model_name}_aa.png')
