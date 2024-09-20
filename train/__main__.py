"""Training script."""

import argparse
import datetime
from itertools import islice
import json
from pathlib import Path

import orbax.checkpoint as ocp
import tomli_w
from tqdm import tqdm

from dataops.io import iter_tasks, read_toml
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
    ts_path = Path('data').resolve() / exp['task_sequence']['name']
    metadata = read_toml(ts_path / 'metadata.toml')
    if 'ood_name' in exp['task_sequence']:
        ood_ts_path = Path('data').resolve() / exp['task_sequence']['ood_name']
        ood_metadata = read_toml(ood_ts_path / 'metadata.toml')

    # prepare directory for checkpoints
    result_path = Path('results').resolve() / args.experiment_id
    ckpt_path = result_path / 'ckpt'
    ckpt_path.mkdir(parents=True, exist_ok=True)
    ocp.test_utils.erase_and_create_empty(ckpt_path)

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
            trainer['name'],
            trainer['immutables']
        ) for trainer in exp['trainers']
    ]

    # train, log and checkpoint
    Path('logs').mkdir(exist_ok=True)
    log_path = (
        Path('logs')
        / f'{args.experiment_id}_'
        f'{datetime.datetime.now().isoformat()}.jsonl'
    )
    with ocp.StandardCheckpointer() as ckpter:
        for trainer_id, name, immutables in tqdm(
            trainers, leave=False, unit='trainer'
        ):
            trainer = getattr(train, name)(model, model_spec, immutables)
            for i, (xs, ys) in enumerate(
                tqdm(
                    iter_tasks(ts_path, 'training'),
                    total=metadata['length'], leave=False, unit='task'
                )
            ):
                for j, state in tqdm(
                    enumerate(trainer.train(xs, ys)),
                    total=immutables['n_epochs'],
                    leave=False, unit='epoch'
                ):
                    predictor = (
                        trainer.predictor(
                            model, model_spec, immutables['n_comp'], state.params
                        ) if isinstance(trainer, train.state.mixins.GSGaussMixin)
                        else trainer.predictor(model, model_spec, state.params)
                    )
                    result = {'trainer_id': trainer_id, 'task_id': i + 1, 'epoch_num': j + 1}
                    for metric in exp['evaluation']['metrics']:
                        result[metric] = [
                            getattr(metrics, metric)(predictor, xs, ys)
                            for xs, ys in islice(iter_tasks(ts_path, 'validation'), i + 1)
                        ]
                    if 'ood_metrics' in exp['evaluation']:
                        for metric in exp['evaluation']['ood_metrics']:
                            result[metric] = [
                                getattr(metrics, metric)(predictor, xs0, xs1)
                                for (xs0, ys0), (xs1, ys1) in zip(
                                    islice(iter_tasks(ts_path, 'validation'), i + 1),
                                    islice(iter_tasks(ood_ts_path, 'validation'), i + 1)
                                )
                            ]
                    with open(log_path, mode='a') as file:
                        print(json.dumps(result), file=file)
                ckpter.save(
                    ckpt_path / f'{trainer_id}_{i + 1}', trainer.state.params
                )


if __name__ == '__main__':
    main()
