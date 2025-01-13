"""Training script."""

import argparse
import datetime
from importlib import import_module
import json
from pathlib import Path

import orbax.checkpoint as ocp
from tqdm import tqdm

from dataops.io import read_task, read_toml
from evaluate import metrics
from feature_extractor import extract_features, train
from models import ModelSpec, NLL
from models import module_map as models_module_map
from train.trainer import module_map as trainer_module_map


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
        train(args.exp_id, exp_spec)
        extract_features(args.exp_id, exp_spec)
        ts_path = Path('data/features').resolve()
    else:
        ts_path = (
            Path('data/prepped').resolve() / exp_spec['task_sequence']['name']
        )
    metadata = read_toml(ts_path / 'metadata.toml')

    # prepare directory for checkpoints
    results_path = Path('results').resolve() / args.exp_id
    ckpt_path = results_path / 'ckpt'
    ckpt_path.mkdir(parents=True, exist_ok=True)

    # create model
    model = getattr(
        import_module(models_module_map[exp_spec['model']['name']]),
        exp_spec['model']['name']
    )(**exp_spec['model']['args'])
    model_spec = ModelSpec(
        nll=NLL[exp_spec['model']['spec']['nll']],
        in_shape=exp_spec['model']['spec']['in_shape'],
        out_shape=exp_spec['model']['spec']['out_shape'],
        cratio=exp_spec['model']['spec']['cratio'],
        cscale=exp_spec['model']['spec']['cscale']
    )

    # train, log and checkpoint
    Path('logs').mkdir(exist_ok=True)
    log_path = (
        Path('logs')
        / f'{args.exp_id}_'
        f'{datetime.datetime.now().isoformat()}.jsonl'
    )
    with ocp.StandardCheckpointer() as ckpter:
        trainer_specs = tqdm(exp_spec['trainers'], leave=False, unit='trainer')
        for trainer_spec in trainer_specs:
            trainer_specs.set_description(
                f'Training for {trainer_spec["id"]}'
            )
            trainer_id = trainer_spec['id']
            trainer_class = getattr(
                import_module(trainer_module_map[trainer_spec['name']]),
                trainer_spec['name']
            )
            train_hparams = trainer_spec['hparams']['train']
            predict_hparams = trainer_spec['hparams']['predict']

            trainer = trainer_class(model, model_spec, train_hparams)
            task_ids = tqdm(
                range(1, metadata['length'] + 1),
                leave=False, unit='task'
            )
            for task_id in task_ids:
                task_ids.set_description(f'Task {task_id}')
                xs, ys = read_task(ts_path, 'training', task_id)
                states = tqdm(
                    trainer.train(xs, ys),
                    total=train_hparams['n_epochs'],
                    leave=False, unit='epoch'
                )
                for epoch_num, state in enumerate(states, start=1):
                    states.set_description(f'Epoch {epoch_num}')
                    predictor = trainer.predictor_class(
                        model, model_spec, predict_hparams, state.params
                    )
                    result = {
                        'trainer_id': trainer_id,
                        'task_id': task_id,
                        'epoch_num': epoch_num
                    }
                    for metric in exp_spec['evaluation']['metrics']:
                        result[metric] = [
                            getattr(metrics, metric)(
                                predictor,
                                *read_task(ts_path, 'validation', i)
                            ) for i in range(1, task_id + 1)
                        ]
                        result[f'average_{metric}'] = (
                            sum(result[metric]) / task_id
                        )
                    with open(log_path, mode='a') as file:
                        print(json.dumps(result), file=file)
                path = ckpt_path / f'{trainer_id}_{task_id}'
                path.mkdir(parents=True, exist_ok=True)
                ocp.test_utils.erase_and_create_empty(path)
                path.rmdir()
                ckpter.save(path, trainer.state.params)


if __name__ == '__main__':
    main()
