"""Evaluation script."""

import argparse
from pathlib import Path

import orbax.checkpoint as ocp

from dataops.io import iter_tasks, read_toml
import models
import train
from train.base import get_pass_size

from .metrics import accuracy


def main():
    """Run the main script."""
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_id', help='experiment ID')
    args = parser.parse_args()

    # read experiment specifications
    exp_path = Path('experiments').resolve()
    exp = read_toml(exp_path / f'{args.experiment_id}.toml')

    # read metadata
    ts_path = Path('data').resolve() / exp['task_sequence']['name']
    metadata = read_toml(ts_path / 'metadata.toml')

    # set checkpoint path
    ckpt_path = Path('results').resolve() / args.experiment_id / 'ckpt'

    # create model and trainers
    model = getattr(models, exp['model']['name'])(**exp['model']['spec'])
    trainers = [
        (
            trainer['id'],
            getattr(train, trainer['name'])(
                model, trainer['immutables'], metadata
            )
        ) for trainer in exp['trainers']
    ]

    # restore checkpoint and evaluate
    pass_size = get_pass_size(metadata['input_shape'])
    result = {}
    with ocp.StandardCheckpointer() as ckpter:
        for i, (trainer_id, trainer) in enumerate(trainers):
            result[trainer_id] = []
            for j in range(metadata['length']):
                trainer.state = trainer.state.replace(params=ckpter.restore(
                    ckpt_path / f'{trainer_id}_{j + 1}',
                    target=trainer.init_state().params
                ))
                predict = trainer.make_predict()
                result[trainer_id].append([
                    accuracy(pass_size, predict, xs, ys)
                    for xs, ys in iter_tasks(ts_path, 'testing')
                ])
    print(result)


if __name__ == '__main__':
    main()
