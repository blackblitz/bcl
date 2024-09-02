"""Training script."""

import argparse
from pathlib import Path

import orbax.checkpoint as ocp
from tqdm import tqdm

from dataops.io import iter_tasks, read_toml
import models
import train


def main():
    """Run the main script."""
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_id', help='experiment ID')
    args = parser.parse_args()

    # read experiment specifications and metadata
    exp_path = (Path('experiments') / args.experiment_id).resolve()
    spec = read_toml(exp_path / 'spec.toml')
    ts_path = (Path('data') / spec['task_sequence']['name']).resolve()
    metadata = read_toml(ts_path / 'metadata.toml')

    # create model and trainers
    model = getattr(models, spec['model']['name'])(**spec['model']['spec'])
    trainers = [
        (
            trainer['id'],
            getattr(train, trainer['name'])(model, trainer['immutables'])
        ) for trainer in spec['trainers']
    ]

    # create checkpointer
    ckpt_path = exp_path / 'ckpt'
    ocp.test_utils.erase_and_create_empty(ckpt_path)
    ckpter = ocp.StandardCheckpointer()

    # train and checkpoint
    for trainer_id, trainer in tqdm(trainers, leave=False, unit='trainer'):
        for j, (xs, ys) in enumerate(
            tqdm(
                iter_tasks(ts_path, 'training'),
                total=metadata['length'], leave=False, unit='task'
            )
        ):
            trainer.train(xs, ys)
            ckpter.save(
                ckpt_path / f'{trainer_id}_{j + 1}', trainer.state.params
            )


if __name__ == '__main__':
    main()
