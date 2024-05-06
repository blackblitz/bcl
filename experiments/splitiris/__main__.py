"""Script for Split Iris."""

import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import numpy as np
from optax import softmax_cross_entropy_with_integer_labels
from tqdm import tqdm

from train.aqc import AutodiffQuadraticConsolidation
from train.ewc import ElasticWeightConsolidation
from train.loss import make_loss_multi_output
from train.nc import NeuralConsolidation
from train.reg import RegularTrainer
from evaluate import accuracy, predict_softmax
from dataio.datasets import memmap_dataset
from dataio.dataset_sequences import accumulate_full
from dataio.dataset_sequences.splitiris import SplitIris
from .models import make_state, nnet, sreg

plt.style.use('bmh')

splitiris_train = SplitIris()
splitiris_test = SplitIris(train=False)
markers = 'ovsPX'
for name, model in zip(tqdm(['sreg', 'nnet'], unit='model'), [sreg, nnet]):
    fig, ax = plt.subplots(figsize=(12, 6.75))
    state_main, state_consolidator = make_state(
        model.Main(), model.Consolidator()
    )
    loss_basic = make_loss_multi_output(
        state_main, softmax_cross_entropy_with_integer_labels
    )
    labels = tqdm([
        'Joint training',
        'Fine-tuning',
        'Elastic Weight Consolidation',
        'Autodiff Quadratic Consolidation',
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
                'precision': 0.1, 'scale': 20.0, 'size': 1024, 'nsteps': 10000,
                'state_consolidator_init': state_consolidator
            },
            loss_basic
        )
    ]
    xs = range(1, len(splitiris_train) + 1)
    for i, (label, trainer) in enumerate(zip(labels, trainers)):
        aa = []
        datasets = tqdm(splitiris_train, leave=False, unit='task')
        for j, dataset in enumerate(
            accumulate_full(datasets) if i == 0 else datasets
        ):
            x, y = memmap_dataset(dataset)
            trainer.train(1000, None, 1024, x, y)
            #trainer.update_loss()
            #trainer.update_state(1000, None, x, y)
            #trainers[4].hyperparams['state_consolidator'] = state_consolidator
            #trainer.update_hyperparams(x, y)
            aa.append(
                np.mean([
                    accuracy(
                        predict_softmax,
                        trainer.state,
                        *memmap_dataset(dataset)
                    ) for dataset in list(splitiris_test)[: j + 1]
                ])
            )
        ax.plot(xs, aa, marker=markers[i], markersize=10, alpha=0.5, label=label)
    ax.set_xticks(xs)
    ax.set_ylim([-0.1, 1.1])
    ax.set_xlabel('Time')
    ax.set_ylabel('Average Accuracy')
    ax.legend()
    fig.savefig(f'plots/splitiris_aa_{name}.png')
