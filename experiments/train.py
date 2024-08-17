"""Training script."""

import argparse
from functools import partial
from pathlib import Path
from importlib import import_module
import tomllib

from flax.training import orbax_utils
from orbax.checkpoint import PyTreeCheckpointer
from tqdm import tqdm

from dataio import dataset_sequences
from dataio.path import clear
from evaluate import predict
import models

parser = argparse.ArgumentParser()
parser.add_argument('experiment_id')
args = parser.parse_args()
path = Path('experiments') / args.experiment_id
with open(path / 'spec.toml', 'rb') as file:
    spec = tomllib.load(file)
dataset_sequence = getattr(
    dataset_sequences, spec['dataset_sequence']['name']
)(**spec['dataset_sequence']['spec'])['training']
model = getattr(models, spec['model']['name'])(**spec['model']['spec'])
trainers = [
    (
        trainer['id'],
        trainer['hyperparams'],
        getattr(import_module('train'), trainer['name']),
        partial(
            getattr(predict, predictor['name']),
            apply=model.apply, **predictor.get('spec', {})
        )
    ) for trainer in spec['trainers']
    for predictor in [trainer['predictor']]
]
clear(path / 'ckpt')
ckpter = PyTreeCheckpointer()
for trainer_id, trainer_hyperparams, make_trainer, make_predictor in tqdm(
    trainers, leave=False, unit='trainer'
):
    trainer = make_trainer(model, make_predictor, trainer_hyperparams)
    for task_id, task in enumerate(
        tqdm(dataset_sequence, leave=False, unit='task'),
        start=1
    ):
        trainer.train(task)
        ckpter.save(
            path.resolve() / f'ckpt/{trainer_id}_{task_id}',
            trainer.state.params,
            save_args=orbax_utils.save_args_from_target(trainer.state.params)
        )
