"""Training script."""

import argparse
import datetime
import json
from pathlib import Path

import orbax.checkpoint as ocp
from tqdm import tqdm

from dataops.io import read_task, read_toml
from evaluate import metrics
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
    ts_path = (
        Path('features').resolve()
        if 'feature_extractor' in exp
        else Path('data').resolve() / exp['task_sequence']['name']
    )
    metadata = read_toml(ts_path / 'metadata.toml')
    if 'ood_name' in exp['task_sequence']:
        if metadata['length'] > 1 or 'feature_extractor' in exp:
            raise ValueError(
                'OOD testing is only for singleton task sequences '
                'without a pre-trained feature extractor'
            )
        ood_ts_path = Path('data').resolve() / exp['task_sequence']['ood_name']

    # prepare directory for checkpoints
    results_path = Path('results').resolve() / args.experiment_id
    ckpt_path = results_path / 'ckpt'
    ckpt_path.mkdir(parents=True, exist_ok=True)

    # create model and trainers
    model = getattr(models, exp['model']['name'])(**exp['model']['args'])
    model_spec = models.ModelSpec(
        nll=models.NLL[exp['model']['spec']['nll']],
        in_shape=exp['model']['spec']['in_shape'],
        out_shape=exp['model']['spec']['out_shape']
    )
    trainers = [
        (
            trainer['id'],
            trainer['name'],
            trainer['immutables']
        ) for trainer in exp['trainers']
    ]

    if model_spec.in_shape != metadata['input_shape']:
        raise ValueError('inconsistent input shapes')

    # train, log and checkpoint
    Path('logs').mkdir(exist_ok=True)
    log_path = (
        Path('logs')
        / f'{args.experiment_id}_'
        f'{datetime.datetime.now().isoformat()}.jsonl'
    )
    with ocp.StandardCheckpointer() as ckpter:
        trainers = tqdm(trainers, leave=False, unit='trainer')
        for trainer_id, name, immutables in trainers:
            trainer = getattr(train, name)(model, model_spec, immutables)
            task_ids = tqdm(
                range(1, metadata['length'] + 1),
                leave=False, unit='task'
            )
            for task_id in task_ids:
                xs, ys = read_task(ts_path, 'training', task_id)
                states = tqdm(
                    trainer.train(xs, ys),
                    total=immutables['n_epochs'],
                    leave=False, unit='epoch'
                )
                for epoch_num, state in enumerate(states, start=1):
                    predictor = (
                        trainer.predictor(
                            model, model_spec,
                            immutables['n_comp'], state.params
                        ) if isinstance(
                            trainer, train.state.mixins.GSGaussMixin
                        ) else trainer.predictor(
                            model, model_spec, state.params
                        )
                    )
                    result = {
                        'trainer_id': trainer_id,
                        'task_id': task_id,
                        'epoch_num': epoch_num
                    }
                    for metric in exp['evaluation']['metrics']:
                        result[metric] = [
                            getattr(metrics, metric)(
                                predictor,
                                *read_task(ts_path, 'validation', i)
                            ) for i in range(1, task_id + 1)
                        ]
                    if 'ood_metrics' in exp['evaluation']:
                        for metric in exp['evaluation']['ood_metrics']:
                            result[metric] = [
                                getattr(metrics, metric)(
                                    predictor,
                                    read_task(ts_path, 'validation', i)[0],
                                    read_task(ood_ts_path, 'validation', i)[0]
                                ) for i in range(1, task_id + 1)
                            ]
                    with open(log_path, mode='a') as file:
                        print(json.dumps(result), file=file)
                path = ckpt_path / f'{trainer_id}_{task_id}'
                ocp.test_utils.erase_and_create_empty(path)
                path.rmdir()
                ckpter.save(path, trainer.state.params)


if __name__ == '__main__':
    main()
