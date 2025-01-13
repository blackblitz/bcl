"""Evaluation script."""

import argparse
from importlib import import_module
import json
from pathlib import Path

from dataops.io import read_task, read_toml
from feature_extractor import extract_features
from models import ModelSpec, NLL
from models import module_map as models_module_map
from train import select_trainers
from train.trainer import module_map as trainer_module_map

from . import metrics


def main():
    """Run the main script."""
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_id', help='experiment ID')
    args = parser.parse_args()

    # read experiment specifications
    exp_path = Path('experiments').resolve()
    exp_spec = read_toml(exp_path / f'{args.exp_id}.toml')

    # read metadata
    if 'feature_extractor' in exp_spec:
        extract_features(args.exp_id, exp_spec)
        ts_path = Path('data/features').resolve()
    else:
        ts_path = (
            Path('data/prepped').resolve() / exp_spec['task_sequence']['name']
        )
    metadata = read_toml(ts_path / 'metadata.toml')

    # set results path
    results_path = Path('results').resolve() / args.exp_id

    # create model and trainers
    model = getattr(
        import_module(models_module_map[exp_spec['model']['name']]),
        exp_spec['model']['name']
    )(**exp_spec['model']['args'])
    mspec = ModelSpec(
        nll=NLL[exp_spec['model']['spec']['nll']],
        in_shape=exp_spec['model']['spec']['in_shape'],
        out_shape=exp_spec['model']['spec']['out_shape'],
        cratio=exp_spec['model']['spec']['cratio'],
        cscale=exp_spec['model']['spec']['cscale']
    )

    # restore checkpoint and evaluate
    Path(results_path / 'evaluation.jsonl').write_bytes(b'')
    for trainer_spec in select_trainers(
        exp_spec, model, mspec, results_path, ts_path, metadata
    ).values():
        trainer_id = trainer_spec['id']
        trainer_class = getattr(
            import_module(trainer_module_map[trainer_spec['name']]),
            trainer_spec['name']
        )
        hparams = trainer_spec['hparams']['predict']

        for task_id in range(1, metadata['length'] + 1):
            path = results_path / f'ckpt/{trainer_id}_{task_id}'
            predictor = trainer_class.predictor_class.from_checkpoint(
                model, mspec, hparams, path
            )
            result = {'trainer_id': trainer_id, 'task_id': task_id}
            for metric in exp_spec['evaluation']['metrics']:
                result[metric] = [
                    getattr(metrics, metric)(
                        predictor,
                        *read_task(ts_path, 'testing', i)
                    ) for i in range(1, task_id + 1)
                ]
                result[f'average_{metric}'] = (
                    sum(result[metric]) / task_id
                )
            with open(results_path / 'evaluation.jsonl', mode='a') as file:
                print(json.dumps(result), file=file)


if __name__ == '__main__':
    main()
