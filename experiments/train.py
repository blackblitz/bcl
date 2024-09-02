"""Training script."""

import argparse
from pathlib import Path

from orbax.checkpoint import PyTreeCheckpointer
from orbax.checkpoint.test_utils import erase_and_create_empty
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
    (exp_path / 'ckpt').mkdir(parents=True, exist_ok=True)
    erase_and_create_empty(exp_path / 'ckpt')
    ckpter = PyTreeCheckpointer()

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
                exp_path / f'ckpt/{trainer_id}_{j + 1}', trainer.state.params
            )


if __name__ == '__main__':
    main()
