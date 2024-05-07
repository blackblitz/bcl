"""Script for Pretrained Split MNIST."""

from importlib.resources import files

import jax.numpy as jnp
from jax import jit, random
import matplotlib.pyplot as plt
import numpy as np
from orbax.checkpoint import PyTreeCheckpointer
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
from dataio.dataset_sequences.splitmnist import SplitMNIST
from .models import make_state, sr
from .pretrain.models import cnnswish, cnntanh, make_state_pretrained

plt.style.use('bmh')



splitmnist_train = SplitMNIST()
splitmnist_test = SplitMNIST(train=False)
markers = 'ovsPX'
for name, model in zip(tqdm(['cnnswish', 'cnntanh'], unit='model'), [cnnswish, cnntanh]):
    params = PyTreeCheckpointer().restore(
        files('experiments.pretrained_splitmnist.pretrain') / name
    )
    state = make_state_pretrained(model.Model()).replace(params=params)
    apply_x = jit(
        lambda x: state.apply_fn(
            {"params": state.params}, x,
            method=lambda m, x: m.feature_extractor(x)
        )
    )
    fig, ax = plt.subplots(figsize=(12, 6.75))
    state_main, state_consolidator = make_state(
        sr.Main(), sr.Consolidator()
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
    xs = range(1, len(splitmnist_train) + 1)
    for i, (label, trainer) in enumerate(zip(labels, trainers)):
        aa = []
        datasets = tqdm(splitmnist_train, leave=False, unit='task')
        for j, dataset in enumerate(
            accumulate_full(datasets) if i == 0 else datasets
        ):
            trainer.train(
                10, 64, 1024,
                *memmap_dataset(dataset, apply_x=apply_x)
            )
            aa.append(
                np.mean([
                    accuracy(
                        predict_softmax, 1024, trainer.state,
                        *memmap_dataset(d, apply_x=apply_x)
                    ) for d in list(splitmnist_test)[: j + 1]
                ])
            )
        ax.plot(xs, aa, marker=markers[i], markersize=10, alpha=0.5, label=label)
    ax.set_xticks(xs)
    ax.set_ylim([-0.1, 1.1])
    ax.set_xlabel('Time')
    ax.set_ylabel('Average Accuracy')
    ax.legend()
    fig.savefig(f'plots/pretrained_splitmnist_aa_{name}.png')
