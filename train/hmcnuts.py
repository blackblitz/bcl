"""HMC-NUTS training script."""

import argparse
import datetime
from importlib import import_module
import json
from multiprocessing import cpu_count
from os import environ
from pathlib import Path

import orbax.checkpoint as ocp
from tqdm import tqdm

ncpus = cpu_count()
environ['JAX_PLATFORM_NAME'] = 'cpu'
environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={ncpus}'

from dataops.io import read_task, read_toml  # noqa: E402
from evaluate import metrics  # noqa: E402
from models import ModelSpec, NLL  # noqa: E402
from models import module_map as models_module_map  # noqa: E402
from train.trainer.sampling import HMCNUTS  # noqa: E402


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

    # create model
    model = getattr(
        import_module(models_module_map[exp['model']['name']]),
        exp['model']['name']
    )(**exp['model']['args'])
    mspec = ModelSpec(
        nll=NLL[exp['model']['spec']['nll']],
        in_shape=exp['model']['spec']['in_shape'],
        out_shape=exp['model']['spec']['out_shape']
    )

    # train, log and checkpoint
    Path('logs').mkdir(exist_ok=True)
    log_path = (
        Path('logs')
        / f'{args.experiment_id}_hmcnuts_'
        f'{datetime.datetime.now().isoformat()}.jsonl'
    )
    with ocp.StandardCheckpointer() as ckpter:
        hparams = {
            'final_only': True, 'n_chains': ncpus, 'n_steps': 100,
            'precision': 1.0, 'seed': 1337
        }
        trainer = HMCNUTS(model, mspec, hparams)
        trainer_id = 'hmcnuts'
        task_ids = tqdm(
            range(1, metadata['length'] + 1),
            leave=False, unit='task'
        )
        for task_id in task_ids:
            xs, ys = read_task(ts_path, 'training', task_id)
            trainer.train(xs, ys)
            predictor = trainer.predictor_class(
                model, mspec, {}, trainer.sample
            )
            result = {
                'trainer_id': trainer_id,
                'task_id': task_id
            }
            for metric in exp['evaluation']['metrics']:
                result[metric] = [
                    getattr(metrics, metric)(
                        predictor,
                        *read_task(ts_path, 'validation', i)
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
                            read_task(ts_path, 'validation', i)[0],
                            read_task(ood_ts_path, 'validation', i)[0]
                        ) for i in range(1, task_id + 1)
                    ]
                    result[f'average_{metric}'] = (
                        sum(result[metric]) / task_id
                    )
            with open(log_path, mode='a') as file:
                print(json.dumps(result), file=file)
            path = ckpt_path / f'{trainer_id}_{task_id}'
            ocp.test_utils.erase_and_create_empty(path)
            path.rmdir()
            ckpter.save(path, trainer.sample)


if __name__ == '__main__':
    main()
