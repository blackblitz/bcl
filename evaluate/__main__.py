"""Evaluation script."""

import argparse
from importlib import import_module
import json
import numpy as np
from pathlib import Path

from dataops.io import read_task, read_toml
from models import ModelSpec, NLL
from models import module_map as models_module_map
from train import module_map as train_module_map
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

    # set results path
    results_path = Path('results').resolve() / args.experiment_id

    # create model and trainers
    model = getattr(
        import_module(models_module_map[exp['model']['name']]),
        exp['model']['name']
    )(**exp['model']['args'])
    model_spec = ModelSpec(
        nll=NLL[exp['model']['spec']['nll']],
        in_shape=exp['model']['spec']['in_shape'],
        out_shape=exp['model']['spec']['out_shape']
    )

    trainer_specs_map = {}
    for trainer_spec in exp['trainers']:
        name = trainer_spec['name']
        trainer_specs_map.setdefault(name, [])
        trainer_specs_map[name].append(trainer_spec)

    for name, trainer_specs in trainer_specs_map.items():
        if len(trainer_specs) > 1:
            validation_scores = np.zeros((len(trainer_specs),))
            for i, trainer_spec in enumerate(trainer_specs):
                trainer_id = trainer_spec['id']
                trainer_class = getattr(
                    import_module(train_module_map[trainer_spec['name']]),
                    trainer_spec['name']
                )
                immutables = trainer_spec['immutables']['predict']
                task_id = metadata['length']
                predictor = trainer_class.predictor_class.from_checkpoint(
                    model, model_spec, immutables,
                    results_path / f'ckpt/{trainer_id}_{task_id}'
                )
                metric = getattr(
                    metrics, exp['evaluation']['validation_metric']
                )
                validation_scores[i] = np.mean([
                    metric(predictor, *read_task(ts_path, 'validation', i))
                    for i in range(1, task_id + 1)
                ])
            trainer_specs_map[name] = trainer_specs[validation_scores.argmax()]

        else:
            trainer_specs_map[name] = trainer_specs[0]

    # restore checkpoint and evaluate
    Path(results_path / 'evaluation.jsonl').write_bytes(b'')
    for trainer_spec in trainer_specs_map.values():
        trainer_id = trainer_spec['id']
        trainer_class = getattr(
            import_module(train_module_map[trainer_spec['name']]),
            trainer_spec['name']
        )
        immutables = trainer_spec['immutables']['predict']

        for task_id in range(1, metadata['length'] + 1):
            path = results_path / f'ckpt/{trainer_id}_{task_id}'
            predictor = trainer_class.predictor_class.from_checkpoint(
                model, model_spec, immutables, path
            )
            result = {'trainer_id': trainer_id, 'task_id': task_id}
            for metric in exp['evaluation']['metrics']:
                result[metric] = [
                    getattr(metrics, metric)(
                        predictor,
                        *read_task(ts_path, 'testing', i)
                    ) for i in range(1, task_id + 1)
                ]
                result[f'average_{metric}'] = (
                    sum(result[metric]) / task_id
                )
            if 'ood_metrics' in exp['evaluation']:
                for metric in exp['evaluation']['ood_metrics']:
                    result[metric] = [
                        getattr(metrics, metric)(
                            predictor,
                            read_task(ts_path, 'testing', i)[0],
                            read_task(ood_ts_path, 'testing', i)[0]
                        ) for i in range(1, task_id + 1)
                    ]
                    result[f'average_{metric}'] = (
                        sum(result[metric]) / task_id
                    )
            with open(results_path / 'evaluation.jsonl', mode='a') as file:
                print(json.dumps(result), file=file)


if __name__ == '__main__':
    main()
