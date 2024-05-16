"""Script for Pre-trained Split CIFAR-10."""

from importlib.resources import files

from jax import jit
import msgpack
from orbax.checkpoint import PyTreeCheckpointer
from tqdm import tqdm

from dataio.datasets import memmap_dataset
from dataio.dataset_sequences import accumulate_full
from dataio.dataset_sequences.split10 import SplitCIFAR10
from evaluate import accuracy
from train import Trainer
from train.qc import (
    AutodiffQuadraticConsolidation,
    ElasticWeightConsolidation,
    SynapticIntelligence
)
from train.nc import NeuralConsolidation

from .models import make_state, sreg
from .pretrain.models import cnnswish, cnntanh, make_state_pretrained

splitcifar10_train = SplitCIFAR10()
splitcifar10_test = SplitCIFAR10(train=False)
labels = [
    'Joint training',
    'Fine-tuning',
    'Elastic Weight Consolidation',
    'Synaptic Intelligence',
    'Autodiff Quadratic Consolidation',
    'Neural Consolidation'
]
trainer_kwargs = {
    'batch_size_hyperparams': 1024, 'batch_size_state': 64, 'n_epochs': 100
}
result = {}
for name, model in zip(
    tqdm(['cnnswish', 'cnntanh'], unit='model'),
    [cnnswish, cnntanh]
):
    result[name] = {label: [] for label in labels}
    params = PyTreeCheckpointer().restore(
        files('experiments.pretrained_splitcifar10.pretrain') / name
    )
    state = make_state_pretrained(model.Model()).replace(params=params)
    apply_x = jit(
        lambda x: state.apply_fn(
            {"params": state.params}, x,
            method=lambda m, x: m.feature_extractor(x)
        )
    )
    state_main, state_consolidator = make_state(
        sreg.Main(), sreg.Consolidator()
    )
    trainers = [
        Trainer(state_main, {'precision': 0.1}, **trainer_kwargs),
        Trainer(state_main, {'precision': 0.1}, **trainer_kwargs),
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
        datasets_train = tqdm(splitcifar10_train, leave=False, unit='task')
        for j, dataset_train in enumerate(
            accumulate_full(datasets_train) if i == 0 else datasets_train
        ):
            trainer.train(*memmap_dataset(dataset_train, apply_x=apply_x))
            result[name][label].append([
                accuracy(
                    True, trainer.state,
                    None, *memmap_dataset(dataset_test, apply_x=apply_x)
                ) for dataset_test in list(splitcifar10_test)[: j + 1]
            ])
with open('results/pretrained_splitcifar10.dat', 'wb') as file:
    file.write(msgpack.packb(result))
