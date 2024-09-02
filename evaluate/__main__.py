"""Evaluation script."""

import argparse
from functools import partial
from pathlib import Path
from importlib import import_module
import tomllib

import numpy as np
from orbax.checkpoint import PyTreeCheckpointer

from dataio.datasets import to_arrays
from evaluate.metrics import accuracy

parser = argparse.ArgumentParser()
parser.add_argument('experiment_id')
args = parser.parse_args()

path = Path('experiments') / args.experiment_id
with open(path / 'spec.toml', 'rb') as file:
    spec = tomllib.load(file)

dataset_sequence = getattr(
    import_module('dataio.dataset_sequences'),
    spec['dataset_sequence']['name']
)(**spec['dataset_sequence']['spec'])['testing']
pass_batch_size = spec['dataset_sequence']['pass_batch_size']
model = getattr(
    import_module(spec['model']['module']),
    spec['model']['name']
)(**spec['model']['spec'])
make_predictors = [
    (
        trainer['id'],
        partial(
            getattr(import_module(predictor['module']), predictor['name']),
            apply=model.apply, **predictor.get('spec', {})
        )
    ) for trainer in spec['trainers']
    for predictor in [trainer['predictor']]
]
ckpter = PyTreeCheckpointer()
result = {}
for i, (trainer_id, make_predictor) in enumerate(make_predictors):
    result[trainer_id] = []
    for j, (task_id, task) in enumerate(enumerate(dataset_sequence, start=1)):
        params = ckpter.restore(
            path.resolve() / f'ckpt/{trainer_id}_{task_id}'
        )
        predictor = make_predictor(params=params)
        acc = []
        for k in range(j + 1):
            xs, ys = to_arrays(dataset_sequence[k])
            acc.append(accuracy(pass_batch_size, predictor, xs, ys))
        result[trainer_id].append(acc)
print(result)
