"""Evaluation script."""

import argparse
from itertools import islice
import json
from pathlib import Path

from dataops.io import iter_tasks, read_toml
import models
import train

from . import metrics


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
    if 'ood_name' in exp['task_sequence']:
        ood_ts_path = Path('data').resolve() / exp['task_sequence']['ood_name']
        ood_metadata = read_toml(ood_ts_path / 'metadata.toml')

    # set results path
    results_path = Path('results').resolve() / args.experiment_id

    # create model and trainers
    model = getattr(models, exp['model']['name'])(**exp['model']['args'])
    model_spec = models.ModelSpec(
        fin_act=models.FinAct[exp['model']['spec']['fin_act']],
        in_shape=exp['model']['spec']['in_shape'],
        out_shape=exp['model']['spec']['out_shape']
    )
    trainers = [
        (
            trainer['id'],
            getattr(train, trainer['name']),
            trainer['immutables']
        ) for trainer in exp['trainers']
    ]

    # restore checkpoint and evaluate
    Path(results_path / 'evaluation.jsonl').write_bytes(b'')
    for trainer_id, trainer, immutables in trainers:
        for i in range(metadata['length']):
            path = results_path / f'ckpt/{trainer_id}_{i + 1}'
            predictor = (
                trainer.predictor.from_checkpoint(
                    model, model_spec, immutables['n_comp'], path
                ) if issubclass(trainer, train.state.mixins.GSGaussMixin)
                else trainer.predictor.from_checkpoint(
                    model, model_spec, path
                )
            )
            result = {'trainer_id': trainer_id, 'task_id': i + 1}
            for metric in exp['evaluation']['metrics']:
                result[metric] = [
                    getattr(metrics, metric)(predictor, xs, ys)
                    for xs, ys in islice(iter_tasks(ts_path, 'testing'), i + 1)
                ]
            for metric in exp['evaluation']['ood_metrics']:
                result[metric] = [
                    getattr(metrics, metric)(predictor, xs0, xs1)
                    for (xs0, ys0), (xs1, ys1) in zip(
                        islice(iter_tasks(ts_path, 'testing'), i + 1),
                        islice(iter_tasks(ood_ts_path, 'testing'), i + 1)
                    )
                ]
            with open(results_path / 'evaluation.jsonl', mode='a') as file:
                print(json.dumps(result), file=file)


if __name__ == '__main__':
    main()
