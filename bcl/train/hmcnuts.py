"""HMC-NUTS training script."""

import argparse
import datetime
from importlib import import_module
import json
from pathlib import Path

import orbax.checkpoint as ocp  # noqa: E402
from tqdm import tqdm  # noqa: E402

from ..dataops.io import read_task, read_toml  # noqa: E402
from ..evaluate import metrics  # noqa: E402
from ..models import ModelSpec, NLL  # noqa: E402
from ..models import module_map as models_module_map  # noqa: E402
from ..train.trainer.sampling import HMCNUTS  # noqa: E402


def main():
    """Run the main script."""
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_id', help='experiment ID')
    parser.add_argument(
        'n_chains', type=int, help='number of chains to run in parallel'
    )
    parser.add_argument(
        '-p', '--precision', type=float, default=1.0, help='initial prior precision'
    )
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
    mspec = ModelSpec(
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
        / f'{args.exp_id}_hmcnuts_'
        f'{datetime.datetime.now().isoformat()}.jsonl'
    )
    with ocp.StandardCheckpointer() as ckpter:
        hparams = {
            'final_only': True, 'n_chains': args.n_chains, 'n_steps': 100,
            'precision': args.precision, 'seed': 1337
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
            ocp.test_utils.erase_and_create_empty(path)
            path.rmdir()
            ckpter.save(path, trainer.sample)


if __name__ == '__main__':
    main()
