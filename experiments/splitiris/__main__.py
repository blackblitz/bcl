"""Script for Split Iris."""

import msgpack
from tqdm import tqdm

from dataio.datasets import memmap_dataset
from dataio.dataset_sequences import accumulate_full
from dataio.dataset_sequences.splitsklearn import SplitIris
from evaluate import accuracy
from train.qc import (
    AutodiffQuadraticConsolidation,
    ElasticWeightConsolidation,
    SynapticIntelligence
)
from train.nc import NeuralConsolidation
from train.reg import RegularTrainer

from .models import make_state, nnet, sreg

splitiris_train = SplitIris()
splitiris_test = SplitIris(train=False)
labels = [
    'Joint training',
    'Fine-tuning',
    'Elastic Weight Consolidation',
    'Synaptic Intelligence',
    'Autodiff Quadratic Consolidation',
    'Neural Consolidation'
]
trainer_kwargs = {
    'batch_size_hyperparams': None, 'batch_size_state': None, 'n_epochs': 1000
}
result = {}
for name, model in zip(tqdm(['sreg', 'nnet'], unit='model'), [sreg, nnet]):
    result[name] = {label: [] for label in labels}
    state_main, state_consolidator = make_state(
        model.Main(), model.Consolidator()
    )
    trainers = [
        RegularTrainer(state_main, {'precision': 0.1}, **trainer_kwargs),
        RegularTrainer(state_main, {'precision': 0.1}, **trainer_kwargs),
        ElasticWeightConsolidation(
            state_main, {'precision': 0.1, 'lambda': 1.0}, **trainer_kwargs
        ),
        SynapticIntelligence(
            state_main, {'precision': 0.1, 'xi': 1.0}, **trainer_kwargs
        ),
        AutodiffQuadraticConsolidation(
            state_main, {'precision': 0.1}, **trainer_kwargs
        ),
        NeuralConsolidation(
            state_main,
            {
                'precision': 0.1, 'radius': 20.0, 'size': 1024, 'nsteps': 1000,
                'state': state_consolidator, 'reg': 0.1
            },
            **trainer_kwargs
        )
    ]
    for i, (label, trainer) in enumerate(
        zip(tqdm(labels, leave=False, unit='algorithm'), trainers)
    ):
        datasets_train = tqdm(splitiris_train, leave=False, unit='task')
        for j, dataset_train in enumerate(
            accumulate_full(datasets_train) if i == 0 else datasets_train
        ):
            trainer.train(*memmap_dataset(dataset_train))
            result[name][label].append([
                accuracy(
                    True, trainer.state,
                    None, *memmap_dataset(dataset_test)
                ) for dataset_test in list(splitiris_test)[: j + 1]
            ])
with open('results/splitiris.dat', 'wb') as file:
    file.write(msgpack.packb(result))
