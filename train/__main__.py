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

    # read experiment specifications
    exp_path = Path('experiments').resolve()
    exp = read_toml(exp_path / f'{args.experiment_id}.toml')

    # read metadata
    ts_path = Path('data').resolve() / exp['task_sequence']['name']
    metadata = read_toml(ts_path / 'metadata.toml')

    # prepare directory for checkpoints
    ckpt_path = Path('results').resolve() / args.experiment_id / 'ckpt'
    ckpt_path.mkdir(parents=True, exist_ok=True)
    ocp.test_utils.erase_and_create_empty(ckpt_path)

    # create model and trainers
    model = getattr(models, exp['model']['name'])(**exp['model']['spec'])
    trainers = [
        (
            trainer['id'],
            getattr(train, trainer['name'])(model, trainer['immutables'])
        ) for trainer in exp['trainers']
    ]

    # train and checkpoint
    with ocp.StandardCheckpointer() as ckpter:
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
