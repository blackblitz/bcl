"""Evaluation script."""

import argparse
from importlib import import_module
import msgpack

from .. import Experiment
from ..dataops.io import read_task
from ..train import select_trainers
from ..train.trainer import module_map as trainer_module_map

from . import metrics


def main():
    """Run the main script."""
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_id', help='experiment ID')
    args = parser.parse_args()

    exp = Experiment(args.exp_id)
    exp.setup(is_eval=True)

    # restore checkpoint and evaluate
    for metricname in exp.spec['evaluation']['metrics']:
        metricfn = getattr(metrics, metricname)
        result = {}
        for trainer_spec in select_trainers(exp).values():
            trainer_id = trainer_spec['id']
            trainer_class = getattr(
                import_module(trainer_module_map[trainer_spec['name']]),
                trainer_spec['name']
            )
            hparams = trainer_spec['hparams']['predict']

            metricvals = [
                [0] * exp.metadata['length']
                for _ in range(exp.metadata['length'])
            ]
            for i in range(exp.metadata['length']):
                path = exp.paths['checkpoint'] / f'{trainer_id}_{i + 1}'
                predictor = trainer_class.predictor_class.from_checkpoint(
                    exp.model, exp.mspec, hparams, path
                )
                for j in range(exp.metadata['length']):
                    metricvals[i][j] = metricfn(
                        predictor,
                        *read_task(exp.paths['data'], 'testing', j + 1)
                    )
            result[trainer_id] = metricvals
        with open(
            exp.paths['result'] / f'{metricname}.dat', mode='wb'
        ) as file:
            msgpack.pack(result, file)

    exp.teardown()


if __name__ == '__main__':
    main()
