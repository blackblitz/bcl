"""Script for Split MNIST."""

import msgpack
from tqdm import tqdm

from dataio.datasets import memmap_dataset
from dataio.dataset_sequences import accumulate_full
from dataio.dataset_sequences.split10 import SplitMNIST
from evaluate import accuracy
from train.qc import (
    ElasticWeightConsolidation,
    SynapticIntelligence
)
from train.nc import NeuralConsolidation
from train.reg import RegularTrainer

from .models import make_state, cnn, fcnn

splitmnist_train = SplitMNIST()
splitmnist_test = SplitMNIST(train=False)
for name, model in zip(tqdm(['fcnn', 'cnn'], unit='model'), [fcnn, cnn]):
    state_main, state_consolidator = make_state(
        model.Main(), model.Consolidator()
    )
    labels = [
        'Joint training',
        'Fine-tuning',
        'Elastic Weight Consolidation',
        'Synpatic Intelligence',
        'Neural Consolidation'
    ]
    trainers = [
        RegularTrainer(state_main, {'precision': 0.1}, True),
        RegularTrainer(state_main, {'precision': 0.1}, True),
        ElasticWeightConsolidation(
            state_main, {'precision': 0.1, 'lambda': 1.0}, True
        ),
        SynapticIntelligence(
            state_main, {'precision': 0.1, 'xi': 1.0}, True
        ),
        NeuralConsolidation(
            state_main,
            {
                'precision': 0.1, 'radius': 20.0, 'size': 1024, 'nsteps': 1000,
                'state_init': state_consolidator, 'reg': 0.1
            },
            True
        )
    ]
    res = {label: [] for label in labels}
    for i, (label, trainer) in enumerate(
        zip(tqdm(labels, leave=False, unit='algorithm'), trainers)
    ):
        datasets = tqdm(splitmnist_train, leave=False, unit='task')
        for j, dataset_train in enumerate(
            accumulate_full(datasets) if i == 0 else datasets
        ):
            trainer.train(10, 64, 256, *memmap_dataset(dataset_train))
            res[label].append([
                accuracy(
                    True, trainer.state,
                    None, *memmap_dataset(dataset_test)
                ) for dataset_test in list(splitmnist_test)[: j + 1]
            ])
    with open(f'results/splitmnist_{name}.dat', 'wb') as file:
        file.write(msgpack.packb(res))
